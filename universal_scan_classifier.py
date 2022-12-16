# -*- coding: utf-8 -*-
import getpass, ipywidgets as ipw, os, json, shlex, io, re, tempfile, subprocess
import pydicom,numpy as np,csv,warnings,pickle,sys,tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from IPython.display import FileLink

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)

#%load_ext autoreload
#%autoreload 2

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from juxnat_lib.xnat_utils import *

"""
Created on Mon Dec 12 15:45:06 2022

@author: mmilchenko
"""
class ScanClassificationModel:
    def __init__(self):
        #define available tags
        self._potentially_supported_tags=['Modality','Manufacturer','StudyDescription','SeriesDescription','ManufacturerModelName',\
        'BodyPartExamined','ScanningSequence','SequenceVariant','MRAcquisitionType',\
        'SequenceName','ScanOptions','SliceThickness','RepetitionTime','EchoTime','InversionTime',\
        'MagneticFieldStrength','NumberOfPhaseEncodingSteps','EchoTrainLength','PercentSampling',\
        'PercentPhaseFieldOfView','PixelBandwidth','AcquisitionMatrix',\
        'FlipAngle','VariableFlipAngleFlag','PatientPosition','PhotometricInterpretation','Rows',\
                                          'Columns','PixelSpacing']
        #single word
        self._singleton_string_tags=['Modality','Manufacturer','ManufacturerModelName','BodyPartExamined',\
                                    'ScanningSequence','SequenceVariant','MRAcquisitionType','SequenceName',\
                                    'MagneticFieldStrength']
        self._singleton_string_tags_xnat=['type','quality','condition','scanner','modality']        
        
        
        #multi-word possible
        self._composite_string_tags=['StudyDescription','SeriesDescription']
        self._composite_string_tags_xnat=['series_description','note']
        
        #single number
        self._singleton_numeric_tags=['SliceThickness','FlipAngle','RepetitionTime','EchoTime',\
                                      'InversionTime','Rows','Columns']        
        self._singleton_numeric_tags_xnat=['frames']
        
        #array of numbers
        self._array_numeric_tags=['AcquisitionMatrix','PixelSpacing']
        
        #all supported tags
        self._supported_tags=sorted(self._singleton_string_tags+self._composite_string_tags+\
            self._singleton_numeric_tags+self._array_numeric_tags)
        
        self._supported_tags_xnat=sorted(self._singleton_string_tags_xnat+\
                                         self._composite_string_tags_xnat+ \
                                         self._singleton_numeric_tags_xnat)
        
        self._selected_tags=[]
        self._selected_fields_xnat=[]
        self._scan_types=[]
        self._nomenclature_name=""
        
    #def get_single_strings(self):
    #   out=[]
    #   for s in self._selected_tags:
    #        if s in self._singleton_string_tags: out=
        
    def tagname_to_group_element(name):
        rep=str(pydicom.tag.Tag(pydicom.datadict.tag_for_keyword(name)))
        rm='(), '
        for ch in rm: rep=rep.replace(ch,'')
        return rep

    def get_selected_fields_xnat(self):
        return self._selected_fields_xnat
        
    def clear_selected_tags(self):
        self._selected_tags.clear()       
    
    def clear_selected_fields_xnat(self):
        self._selected_fields_xnat.clear()
        
    def check_validity(self):
        '''
        validate input
        '''        
        if len(self._scan_types)<2:
            print('Enter at least two scan types.')
            return False
        if len(self._selected_tags)+len(self._selected_fields_xnat)<1:
            print('Select at least one DICOM tag or XNAT field.')
            return False
        if len(self._nomenclature_name)<1:
            print('Nomenclature name cannot be empty.')
        return True
    
    def load(self,d:json):
        self._scan_types=d['scan_types']
        self._selected_tags=d['selected_dcm_tags']
        self._selected_fields_xnat=d['selected_fields_xnat']
        self._nomenclature_name=d['nomenclature_name']
        self._dense_layer_size=len(self._scan_types)*2


class UniversalScanClassifier:
    def __init__(self,scm:ScanClassificationModel):
        self.classifier=[]
        self.vectorizer=[]
        self._class_vectorizer=None
        self._scm=scm
        #important: must be in aphpabetical order for vectorizer to work correctly.
        self._classes=scm._scan_types        
        #self._scan_list=[]
        
    def load_json(self, json_file):
        with open(json_file, 'r') as fp:
            out_dict=json.loads(fp.read())
        return out_dict
    def save_json(self, var, file):
        with open(file,'w') as fp:
            json.dump(var, fp)
    
    def write_scans_csv(self, scans, file):
        with open(file, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, scans[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(scans)
            
    def read_scans_csv(self, file):
        with open(file,'r') as inf:
            reader = csv.DictReader(inf)
            scans=[{k: str(v) for k,v in row.items()} 
                      for row in csv.DictReader(inf,skipinitialspace=True)]
        return scans
       
    '''
    Create vocabulary from the bag of words. These will act as features.
    '''
    def gen_vocabulary(self,scans):        
        descs=self.prepare_descs(scans)        
        vectorizer=CountVectorizer(min_df=0)
        vectorizer.fit(descs)
        self.vectorizer=vectorizer
        print('the length of vocabulary is ',len(vectorizer.vocabulary_))
     
    #for logreg/svm output, categorical labels are stored as strings
    def prepare_training_vectors(self,scans):
        #labels vector.
        vectorized_descs=self.gen_bow_vectors(scans)
        y=[ s['hof_id'] for s in scans ]
        return vectorized_descs,y
    
    #for a NN output, categorical labels are stored as BOW over vocabulary of class labels.
    def prepare_training_vectors_nn(self,scans,gen_hofids=True):
        if self._class_vectorizer is None:
            vectorizer=CountVectorizer(min_df=0)
            vectorizer.fit(self._classes)
            self._class_vectorizer=vectorizer
        vectorizer=self._class_vectorizer
        vectorized_descs=self.gen_bow_vectors(scans)
        hofids=[ s['hof_id'] for s in scans ] if gen_hofids else []
        return vectorized_descs,vectorizer.transform(hofids).toarray()
    
    def get_coded_tag_value(self,scan,selected_tag):
        s,st=scan,selected_tag
        out=''
        if st in self._scm._singleton_string_tags:
            out=s[st]
        elif st in self._scm._composite_string_tags:
            out=re.sub('[^0-9a-zA-Z ]+',' ',s[st]).split()
        #elif st in self._scm._singleton_numeric_tags:
        #    return str(s[st])
        else:
            out=str(s[st])
        return out
        
    def get_coded_xnat_field_value(self,scan,selected_tag):
        
        #TODO: unique word prefix mapping for single and array numeric values.
        #For that, define a) transform (none, log) and b) prefix-maybe, no need for prefix?
        #then define a) transform and b)discrete sampling step. Prefix is the name of tag.
        
        s,st=scan,selected_tag
        out=""
        if st in self._scm._singleton_string_tags_xnat:
            out=s[st]
        elif st in self._scm._composite_string_tags_xnat:
            out=re.sub('[^0-9a-zA-Z ]+',' ',s[st]).split()
        elif st=='frames':
            try:
                frames='frames{}'.format(str(int(np.around(np.log(1.0+float(s['frames']))*3.0))))
            except:
                frames='frames0'
            out=frames
        #elif st in self._scm._singleton_numeric_tags:
        #    return str(s[st])
        else:
            out=str(s[st])
        return out
        
        
    def prepare_desc(self,scan):
        s=scan
        words=[]
        for st in self._scm._selected_tags:
            words.append(self.get_coded_tag_value(s,st))
        for sf in self._scm._selected_fields_xnat:
            words.append(self.get_coded_xnat_fields_value(s,sf))
        return ' '.join([w for w in words if ((not w.isdigit()) and (len(w)>1)) ])            
'''        
            if st in self._scm._singleton_string_tags:
               desc.append(s[st])
            elif st in self._scm._composite_string_tags:
                desc=(re.sub('[^0-9a-zA-Z ]+',' ',s['series_description'])).split()
                
        
        desc=(re.sub('[^0-9a-zA-Z ]+',' ',s['series_description'])).split()
        #compressed representation of the number of frames.
        try:
            frames='frames{}'.format(str(int(np.around(np.log(1.0+float(s['frames']))*3.0))))
        except:
            frames='frames0'
        desc.append(frames)
'''

    def prepare_descs(self,scans):
        #descs are 'sentences' that contain all supported tags and xnat fields.
        #(former series description and log-compressed number of frames.)
        return [prepare_desc(s) for s in scans ]
        
    def gen_bow_vectors(self,scans):
        if not self.vectorizer: return []
        descs=self.prepare_descs(scans)
        return self.vectorizer.transform(descs).toarray()
    
    def train_nn(self,X,y,test_split,epochs=10,batch_size=10):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_split,random_state=1000)
        input_dim=X_train.shape[1]
        print('input_dim:',input_dim)
        model = Sequential()
        model.add(layers.Dense(self._scm._dense_layer_size,input_dim=input_dim,activation='relu'))
        #model.add(layers.Dense(36,input_dim=input_dim,activation='relu'))        
        #model.add(layers.Dense(18,activation='relu'))
        model.add(layers.Dense(len(self._classes),activation='sigmoid'))
        print('output_dim:',len(self._classes))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy','categorical_accuracy'])
        model.summary()
        self.classifier=model
        #self.classifier.fit(X_train,y_train,epochs=10,verbose=True,validation_data=(X_test,y_test),batch_size=10)
        hist=self.classifier.fit(X_train,y_train,epochs=epochs,verbose=True,validation_data=(X_test,y_test),batch_size=batch_size)
        self.plot_nn_train_history(hist)
        
    def plot_nn_train_history(self,history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        
    def infer_nn(self,scans):
        vecs,ids=self.prepare_training_vectors_nn(scans,False)
        y_fit=self.classifier.predict(vecs)        
        hofids=[ self._classes[np.argmax(y_fit[i])] for i in range(len(y_fit)) ]
        return hofids        
        
    def train_classifier(self,X,y,test_split):
        descs_train,descs_test,y_train,y_test=train_test_split(X,y,test_size=test_split,random_state=1000)
        #classifier=LogisticRegression()
        classifier=LinearSVC()
        #classifier=SVC()
        classifier.fit(descs_train,y_train)
        scoreTest=classifier.score(descs_test,y_test)
        scoreTrain=classifier.score(descs_train,y_train)
        print('Test accuracy:', scoreTest, " train accuracy:",scoreTrain)        
        self.classifier=classifier
        
        return classifier
        
    def _predict_classifier(self,X):
        if not self.classifier: return []
        return self.classifier.predict(X)
        
    def predict_classifier(self, scans):
        vectorized_descs=self.gen_bow_vectors(scans)
        labels=self._predict_classifier(vectorized_descs)
        for i,s in enumerate(scans):
            s['scan_classifier_detected_type']=labels[i]
        return scans
    
    def is_valid_model(self):
        return (self.vectorizer and self.classifier)    
        
    def save_model_nn(self,rt):
        pickle.dump(self.vectorizer,open(rt+'.vec','wb'))
        self.classifier.save(rt+'.hd5')
        
    def load_model_nn(self,rt):
        return self.load_model_nn1(rt+'.vec',rt+'.hd5')
        
    def load_model_nn1(self,model_file,vec_file):
        self.vectorizer=pickle.load(open(vec_file,'rb'))
        self.classifier=tf.keras.models.load_model(model_file)
        return self.vectorizer is not None and self.classifier is not None
    
    def save_model(self, file):
        pickle.dump([self.vectorizer,self.classifier],open(file,'wb'))
                    
    def load_model(self, file):
        self.vectorizer,self.classifier=pickle.load(open(file,'rb'))    
