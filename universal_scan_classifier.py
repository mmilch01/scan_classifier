# -*- coding: utf-8 -*-
import getpass, ipywidgets as ipw, os, json, shlex, io, re, tempfile, subprocess,unittest, argparse
import pydicom,numpy as np,csv,warnings,pickle,sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from pathlib import Path
from matplotlib import pyplot as plt
from zipfile import ZipFile

#%load_ext autoreload
#%autoreload 2

from juxnat_lib import xnat_utils


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
        'PercentPhaseFieldOfView','PixelBandwidth','AcquisitionMatrix','ImageType',\
        'FlipAngle','VariableFlipAngleFlag','PatientPosition','PhotometricInterpretation','Rows',\
                                          'Columns','PixelSpacing','ContrastBolusVolume','ContrastBolusTotalDose',\
                                         'ContrastBolusIngredient','ContrastBolusIngredientConcentration']
        #single word
        self._singleton_string_tags=['Modality','Manufacturer','ManufacturerModelName','BodyPartExamined',\
                                    'ScanningSequence','SequenceVariant','MRAcquisitionType','SequenceName',\
                                    'MagneticFieldStrength','ImageType','Radiopharmaceutical']
        self._singleton_string_tags_xnat=['type','quality','condition','scanner','modality','ID']
        
        
        #multi-word possible
        self._composite_string_tags=['StudyDescription','SeriesDescription']
        self._composite_string_tags_xnat=['series_description','note']
        
        #single number
        self._singleton_numeric_tags=['SliceThickness','FlipAngle','RepetitionTime','EchoTime',\
                                      'InversionTime','Rows','Columns','ContrastBolusVolume','ContrastBolusTotalDose',\
                                         'ContrastBolusIngredient','ContrastBolusIngredientConcentration']
        self._singleton_numeric_tags_xnat=['frames']
        self._sequence_tags=['RadiopharmaceuticalInformationSequence']
        
        #array of numbers
        self._array_numeric_tags=['AcquisitionMatrix','PixelSpacing']
        
        #all supported tags
        self._supported_tags=sorted(self._singleton_string_tags+self._composite_string_tags+\
            self._singleton_numeric_tags+self._array_numeric_tags+self._sequence_tags)
        
        self._supported_tags_xnat=sorted(self._singleton_string_tags_xnat+\
                                         self._composite_string_tags_xnat+ \
                                         self._singleton_numeric_tags_xnat)
        
        self._selected_tags=[]
        self._selected_fields_xnat=[]
        self._scan_types=[]
        self._nomenclature_name=""
        self._classifier_type="perceptron_nn"
        self.verbosity=1
        
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
        if len(self._scan_types)<3 and self._classifier_type=="perceptron_nn":
            print('The perceptron_nn does not support two classes. Please use linear_svm.')
            return False                  
        if len(self._selected_tags)+len(self._selected_fields_xnat)<1:
            print('Select at least one DICOM tag or XNAT field.')
            return False
        if len(self._nomenclature_name)<1:
            print('Nomenclature name cannot be empty.')
        return True
    
    def load(self,d:json):
        self._scan_types=sorted(d['scan_types'])
        
        self._selected_tags=d['selected_dcm_tags']
        self._selected_fields_xnat=d['selected_fields_xnat']
        self._nomenclature_name=d['nomenclature_name']        
        self._dense_layer_size=max(8,len(self._scan_types)*2)
        try:
            self._classifier_type=d['classifier_type']
        except Exception as e:
            self._classifier_type='perceptron_nn'
        
    def load_from_file(self,file):
        try:
            with open(file,'r') as f:
                d=json.load(f)
            self.load(d)
        except Exception as e:
            print(e)
            print('cannot load classification from file:',file)
            return False
        return True
    
class ScanProcessor:
    def __init__(self,scm:ScanClassificationModel):
        self._scm=scm        
        
        
    def filter_unique_dicts(self,dict_list,keys):
        '''
        Функция filter_unique_dicts принимает на вход список dict_list словарей и список keys,
        содержащий поля, по которым нужно произвести фильтрацию. Для каждого словаря в списке 
        словарей функция вычисляет комбинацию значений этих полей, представленную в виде кортежа.
        Эта комбинация затем сравнивается со всеми уже встреченными ранее комбинациями. Если
        текущая комбинация еще не встречалась, то словарь добавляется в список уникальных словарей
        '''
        unique_dicts = {}
        for dictionary in dict_list:
            key_values = tuple(dictionary[key] for key in keys)
            if key_values not in unique_dicts:
                unique_dicts[key_values]={'__usc_occurrences': 1, **dictionary}
            else:
                unique_dicts[key_values]['__usc_occurrences'] =int(unique_dicts[key_values]['__usc_occurrences'])+1
        return list(unique_dicts.values())        
        
    def compress_scans(self,scans):
        '''
        Remove duplicate scans with fields matching the 
        member ScanClassificationModel
        '''
        scm=self._scm
        return self.filter_unique_dicts(scans,scm._selected_fields_xnat + scm._selected_tags)

    def expand_dicts(self,dict_list):
        return [d for d in dict_list for i in range(int(d['__usc_occurrences']))]
    
    def uncompress_scans(self,scans):
        '''
        Restore duplicate scans removed by compress_scans
        Note that fields not in ScanClassificationModel are 
        not restored.
        '''
        if '__usc_occurrences' in scans[0]: return self.expand_dicts(scans)
        else: return scans
        
class UniversalScanClassifier:
    def __init__(self,scm:ScanClassificationModel):
        self.classifier=[]
        self.vectorizer=[]
        self._class_vectorizer=None
        self._scm=scm
        self.verbosity=1
        
        #important: must be in aphpabetical order for vectorizer to work correctly.
        #self._classes=scm._scan_types        
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
        vectorizer=CountVectorizer(min_df=0.0)
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
            vectorizer=CountVectorizer(min_df=0.0)
            vectorizer.fit(self._scm._scan_types)
            self._class_vectorizer=vectorizer
        vectorizer=self._class_vectorizer
        vectorized_descs=self.gen_bow_vectors(scans)
        hofids=[ s['hof_id'] for s in scans ] if gen_hofids else []
        return vectorized_descs,vectorizer.transform(hofids).toarray()
    
    def get_coded_tag_value(self,scan,selected_tag):
        s,st=scan,selected_tag
        out=''
        if st in self._scm._singleton_string_tags:
            out=[st+'_'+s[st]]
        elif st in self._scm._composite_string_tags:
            try:
                out1=re.sub('[^0-9a-zA-Z ]+',' ',s[st]).split()
                out=[ st+'_'+ w for w in out1 ]
            except Exception as e:                
                #print('Exception:',e)
                #print('tag:',st)
                #print('scan:',scan)
                out=[st+'_NA']
                
        #elif st in self._scm._singleton_numeric_tags:            
        #    return str(s[st])
        elif st=='SliceThickness':     
            sout='NA'
            try:
                if float(s[st])<=0.1: sout='NA'
                elif float(s[st])<1.5: sout='high'
                elif float(s[st])<2.5: sout='medium'
                else: sout='low'                
            except Exception as e:
                pass
            out=[st+'_'+sout]
                #print('Exception:',e)
                #print('Value:',s[st])
        else:
            out=[st+'_'+str(s[st])]
        return out
    
    
    def valid_word(self,w):
        return (not w.isdigit()) and (len(w)>1)
    
    def get_coded_xnat_field_value(self,scan,selected_tag):
        
        #TODO: unique word prefix mapping for single and array numeric values.
        #For that, define a) transform (none, log) and b) prefix-maybe, no need for prefix?
        #then define a) transform and b)discrete sampling step. Prefix is the name of tag.
        
        s,st=scan,selected_tag
        out=""
        if st in self._scm._singleton_string_tags_xnat:
            out=[st+'_'+s[st]] if self.valid_word(s[st]) else []
        elif st in self._scm._composite_string_tags_xnat:
            out1=re.sub('[^0-9a-zA-Z ]+',' ',s[st]).split()
            out=[st+'_'+w for w in out1 if self.valid_word(w) ]
        elif st=='frames':
            try:
                frames='frames_{}'.format(str(int(np.around(np.log(1.0+float(s['frames']))*3.0))))
            except:
                frames='frames_0'
            out=[frames]
        #elif st in self._scm._singleton_numeric_tags:
        #    return str(s[st])
        else:
            out=[st+'_'+str(s[st])]
        return out
        
        
    def prepare_desc(self,scan):
        s=scan
        words=[]
        for st in self._scm._selected_tags:
            words+=self.get_coded_tag_value(s,st)
        for sfx in self._scm._selected_fields_xnat:
            words+=self.get_coded_xnat_field_value(s,sfx)
            
        if (self.verbosity>1):
            print('Description words:',words)
           
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
    
    def labeling_valid(self,scans):
        labels=self._scm._scan_types[:]
        labels_not_found=self._scm._scan_types[:]
        valid=True
        for s in scans:
            label=s['hof_id']
            if not label in labels:
                print('invalid label "{}", for scan {})'.format(label, s))
                valid=False
            if label in labels_not_found:
                labels_not_found.remove(label)
        if len(labels_not_found)>0:
            print('The following labels were not found: '+','.join(labels_not_found))
            valid=False
        return valid
        
    def init_and_run_training(self,scans,test_split=0.11,epochs=10,batch_size=10,random_state=1000):
        if self._scm._classifier_type=="perceptron_nn":
            return self.init_and_run_nn_training(scans,test_split=test_split,\
                                                 epochs=eporchs,batch_size=batch_size,\
                                                 random_state=random_state)
        elif self._scm._classifier_type=="linear_svm":
            return self.init_and_run_svm_training(scans,test_split=test_split,random_state=random_state)
    
    def init_and_run_nn_training(self,scans,test_split=0.11,epochs=10,batch_size=10,random_state=1000):
        '''
        Inits and runs NN training from the given set of scans.
        '''        
        print('Checking labeling validity...')
        if not self.labeling_valid(scans):
            print('Invalid labeling, cannot train. Either fix labels or remove unlabeled records from the training set.')
            return False
        
        print('Generating vocabulary...')
        self.gen_vocabulary(scans)
        print('Preparing training vectors...')
        descs,y=self.prepare_training_vectors_nn(scans)
        print('Training...')
        self.train_nn(descs,y,test_split=test_split,epochs=epochs,\
                      batch_size=batch_size,random_state=random_state)
        print('Done.')
        return True
        
    def init_and_run_svm_training(self,scans,test_split=0.11,random_state=1000):
        print('Checking labeling validity...')
        if not self.labeling_valid(scans):
            print('Invalid labeling, cannot train. Either fix labels or remove unlabeled records from the training set.')
            return False

        print('Generating vocabulary...')
        self.gen_vocabulary(scans)
        print('Preparing training vectors...')
        descs,y=self.prepare_training_vectors(scans)
        print('Training...')
        self.train_classifier(descs,y,test_split=test_split,random_state=random_state)
        print('Done.');
        
    def prepare_descs(self,scans):
        #descs are 'sentences' that contain all supported tags and xnat fields.
        #(former series description and log-compressed number of frames.)
        return [self.prepare_desc(s) for s in scans ]
        
    def gen_bow_vectors(self,scans):
        if not self.vectorizer: return []
        descs=self.prepare_descs(scans)
        return self.vectorizer.transform(descs).toarray()
    
    def train_nn(self,X,y,test_split=0.11,epochs=10,batch_size=10, random_state=1000):
        print('test dataset split ratio: {}, epochs: {}, batch size: {}, random state: {}, \
            dense nodes: {}'.format(test_split,epochs,batch_size,random_state,self._scm._dense_layer_size))
              
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_split,random_state=random_state)
        input_dim=X_train.shape[1]
        print('input_dim:',input_dim)
        model = Sequential()
        model.add(layers.Dense(self._scm._dense_layer_size,input_dim=input_dim,activation='relu'))
        #model.add(layers.Dense(36,input_dim=input_dim,activation='relu'))        
        #model.add(layers.Dense(18,activation='relu'))
        model.add(layers.Dense(len(self._scm._scan_types),activation='sigmoid'))
        print('output_dim:',len(self._scm._scan_types))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy','categorical_accuracy'])
        model.summary()
        self.classifier=model
        #self.classifier.fit(X_train,y_train,epochs=10,verbose=True,validation_data=(X_test,y_test),batch_size=10)
        hist=self.classifier.fit(X_train,y_train,epochs=epochs,verbose=True,validation_data=(X_test,y_test),batch_size=batch_size)
        #return hist
        self.plot_nn_train_history(hist)
        
    def plot_nn_train_history(self,history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
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
    
    def infer(self,scans):
        if self._scm._classifier_type=="perceptron_nn":
            return self.infer_nn(scans)        
        elif self._scm._classifier_type=="linear_svm":
            return self.infer_svm(scans)
        else: return None
    
    def infer_nn(self,scans):
        vecs,ids=self.prepare_training_vectors_nn(scans,False)
        y_fit=self.classifier.predict(vecs)        
        hofids=[ self._scm._scan_types[np.argmax(y_fit[i])] for i in range(len(y_fit)) ]
        return hofids

    def infer_nn_ext(self,scans):
        vecs,ids=self.prepare_training_vectors_nn(scans,False)
        pred=self.classifier.predict(vecs)
        print('pred:',pred)
        pred_ord=np.argsort(-pred,axis=1)
        print('pred_ord:',pred_ord)
        pred_inv0=[ self._scm._scan_types[pred_ord[i,0]] for i in range(len(pred)) ]
        pred_inv1=[ self._scm._scan_types[pred_ord[i,1]] for i in range(len(pred)) ]

        max_len=min(len(pred),10)
        print('pred_inv0, first 10:',pred_inv0[:max_len])
        print('pred_inv1, first 10:',pred_inv1[:max_len])
        
        pred_class1=[]
        pred_prob1=[]
        #predicted second most likely class
        pred_prob2=[]
        pred_class2=[]
        pred_entropy=[]
        pred_gini_impurity=[]
        pred_margin_confidence=[]
        series_descriptions=[]
        
        for i in range (0,len(pred_inv0)):
            pred_cur=pred[i]
            pred_ord_cur=pred_ord[i]
            pred_class1+=[ pred_inv0[i] ]
            pred_prob1+=[ pred_cur[pred_ord_cur[0]] ]
            pred_class2+=[ pred_inv1[i] ]
            pred_prob2+=[ pred_cur[pred_ord_cur[1]] ]

            pred_gini_impurity+=[1-np.sum(np.array([pred_cur[i]*pred_cur[i] for i in range (0,len(pred_cur))]))]            
            pred_margin_confidence+=[pred_cur[pred_ord_cur[0]]-pred_cur[pred_ord_cur[1]]]

            try:
                series_descriptions+=[scans[i]['SeriesDescription'].replace(' ','_')+' ']
            except Exception as e:
                series_descriptions+=['NA']
                print('no series description for file',i)
		
        print('First 10 predicted labels:',pred_class1[:max_len])
        return pred_class1,pred_prob1,pred_class2,pred_prob2,pred_gini_impurity,pred_margin_confidence,series_descriptions
    
    def infer_svm(self,scans):
        vectorized_descs=self.gen_bow_vectors(scans)
        return self._predict_classifier(vectorized_descs)
        
    def train_classifier(self,X,y,test_split,random_state=1000):
        descs_train,descs_test,y_train,y_test=train_test_split(X,y,test_size=test_split,random_state=random_state)
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
        
    def save_model_nn(self,file):
        rt=os.path.splitext(file)[0]
        zf,vec,hd5=rt+'.zip',rt+'.vec',rt+'.hd5'
        pickle.dump(self.vectorizer,open(vec,'wb'))
        self.classifier.save(hd5,save_format='h5')
        with ZipFile(zf,'w') as zipfile:
            zipfile.write(hd5,arcname=os.path.basename(hd5))
            zipfile.write(vec,arcname=os.path.basename(vec))            
        os.remove(vec)
        os.remove(hd5)
        
    def load_model_nn(self,zipfile):
        
        #zipfile_dir=os.path.dirname(zipfile)
        zipfile_dir=tempfile.gettempdir()
       	with ZipFile(zipfile,'r') as zf:
            zf.extractall(zipfile_dir)
            
        rt=zipfile_dir+'/'+Path(zipfile).stem
	#rt=os.path.splitext(zipfile)[0]
        vec_file,model_file=rt+'.vec',rt+'.hd5'
        
        self.vectorizer=pickle.load(open(vec_file,'rb'))
        self.classifier=tf.keras.models.load_model(model_file)
        os.remove(vec_file)
        os.remove(model_file)
        return self.vectorizer is not None and self.classifier is not None
    
    def save_model(self, file):
        if self._scm._classifier_type=="perceptron_nn":
            return self.save_model_nn(file)
        elif self._scm._classifier_type=="linear_svm":
            pickle.dump([self.vectorizer,self.classifier],open(file,'wb'))
            
    def load_model(self, file):
        if self._scm._classifier_type=="perceptron_nn":
            return self.load_model_nn(file)
        elif self._scm._classifier_type=="linear_svm":
            self.vectorizer,self.classifier=pickle.load(open(file,'rb'))
            return self.vectorizer is not None and self.classifier is not None
        else:
            return False

    def scans_from_files(self,file_list,tags=None):
        if tags is None: tags=self._scm._supported_tags
        #print('tags:',tags)
        scans=[]
        for file in file_list:
            d=dict()
            ds=pydicom.filereader.dcmread(file,stop_before_pixels=True,specific_tags=tags)
            #ds=pydicom.filereader.dcmread(file,stop_before_pixels=True)
            #print(ds)
            for tag in tags:
                try:
                    if tag=='Radiopharmaceutical':
                        d[tag]=ds['RadiopharmaceuticalInformationSequence'][0]['Radiopharmaceutical'].value
                    else:
                        d[tag]=ds[tag].value
                except Exception as e:
                    pass
            scans+=[d]
        return scans
        
    def classify_dicom_scans(self, out_file):
        '''
            tokenizer_file is not used but is defined for upward compatibility.
        '''
        scans=self.scans_from_files(dicom_files)
        labels1,probs1,labels2,probs2,pred_gini_impurity,pred_margin_confidence,series_descriptions=self.infer_nn_ext(scans)          
        d={'files':dicom_files,'labels1':labels1,'probs1':probs1,'labels2':labels2,'probs2':probs2,'series_descriptions':series_descriptions,\
           'pred_gini_impurity':pred_gini_impurity,'pred_margin_confidence':pred_margin_confidence}
        with open(out_file,mode='w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=d.keys())
            w.writeheader()
            for row in zip(*d.values()): w.writerow(dict(zip(d.keys(),row)))



class UniversalScanClassifierTest:
#    def setUp(self):
#        pass
    def __init__(self):
        self.scm=ScanClassificationModel()
        self.usc=UniversalScanClassifier(self.scm)
        pass
    
    def test_load_nomenclature1(self):
        print('loading from file:',self.scm.load_from_file('./test/neuro_onc.json')==True)
        
    def test_load_nomenclature2(self):
        print('loading from file:',self.scm.load_from_file('./test/mri_types.json')==True)
        
            
    def test_load_training_set1(self):
        try:
            self.scans=self.usc.read_scans_csv('./test/all_scans_hofid.csv')
            if len(self.scans)>0: 
                print ('loaded scans from','./test/all_scans_hofid.csv')
        except Exception as e:
            print(e)
            print('loading scans from file failed')
            
    def test_load_training_set2(self):
        self.scans=self.usc.read_scans_csv('./test/all_scans_function.csv')

    def test_load_training_set3(self):
        self.scans=self.usc.read_scans_csv('./test/all_scans_voxelres_hofid.csv')

    
    def test_train_model1_nn(self,test_split=0.11,epochs=10,batch_size=10,random_state=1000):
        #try:      
        self.test_load_nomenclature1()
        self.test_load_training_set1()
        self.usc.init_and_run_nn_training(self.scans,test_split=test_split,\
                                          epochs=epochs,batch_size=batch_size,random_state=random_state)
        
        self.usc.save_model_nn('./test/neuro-onc-test.zip')
        #except Exception as e:
        #    print('Exception:',e)
        
    def test_train_model1_svm(self,test_split=0.11,random_state=1000):
        self.test_load_nomenclature1()
        self.test_load_training_set1()
        self.usc.init_and_run_svm_training(self.scans,test_split=test_split,random_state=random_state)        
        self.usc.save_model('./test/neuro-onc-test_svm.pkl')
        
    def test_train_model2_svm(self,test_split=0.11,random_state=1000):
        self.test_load_nomenclature2()
        self.test_load_training_set2()
        self.usc.init_and_run_svm_training(self.scans,test_split=test_split,random_state=random_state)        
        self.usc.save_model('./test/mri_types-test_svm.pkl')
        
    def test_train_model2_nn(self,test_split=0.11,epochs=10,batch_size=10,random_state=1000):
        self.test_load_nomenclature2()
        self.test_load_training_set2()
        self.scans=self.usc.read_scans_csv('./test/all_scans_function.csv')
        print ('loaded scans from','./test/all_scans_function.csv')
        self.usc.init_and_run_nn_training(self.scans,test_split=test_split,\
                                          epochs=epochs,batch_size=batch_size,random_state=random_state)
        self.usc.save_model_nn('./test/mri_types-test.zip')

    def test_train_model3_nn(self,test_split=0.11,epochs=10,batch_size=10,random_state=1000):
        #self.test_load_nomenclature3()
        #self.test_load_training_set3()
        self.scm.load_from_file('./test/neuro_onc-dcm.json')
        self.scans=self.usc.read_scans_csv('./test/all_scans_voxelres_hofid.csv')
        print ('loaded scans from','./test/all_scans_voxelres_hofid.csv')
        self.usc.init_and_run_nn_training(self.scans,test_split=test_split,\
                                          epochs=epochs,batch_size=batch_size,random_state=random_state)        
        self.usc.save_model_nn('./test/neuro-onc-test3.zip')

    def test_train_model4_nn(self,test_split=0.11,epochs=10,batch_size=10,random_state=1000):
        #self.test_load_nomenclature3()
        #self.test_load_training_set3()
        self.scm.load_from_file('./test/neuro_onc-dcm.json')
        self.scans=self.usc.read_scans_csv('./test/manual_label_based_on_classification_output_model_fc_39374-600.03.20.2024_2024Apr06_113650.csv')
        print ('loaded scans from','./test/manual_label_based_on_classification_output_model_fc_39374-600.03.20.2024_2024Apr06_113650.csv')
        self.usc.init_and_run_nn_training(self.scans,test_split=test_split,\
                                          epochs=epochs,batch_size=batch_size,random_state=random_state)        
        self.usc.save_model_nn('./test/neuro-onc-test4.zip')

    
    def prediction_accuracy(self,labeled_scans,classified_types):
        scans=labeled_scans
        n=0
        for i in range(len(scans)):
            if classified_types[i]!=scans[i]['hof_id']:
                print('position: {}, predicted: {}, actual: {}'\
                      .format(i,classified_types[i],scans[i]['hof_id']))
                n+=1
        print('Classification accuracy:',1.-n/len(scans))
        print("Done.")
        
    
    def test_validate_model1_svm(self):
        self.test_load_nomenclature1()
        self.usc.load_model('./test/neuro-onc-test_svm.pkl')
        scans=self.usc.read_scans_csv('./test/all_scans_hofid.csv')
        classified_types=self.usc.infer_svm(scans)
        self.prediction_accuracy(scans,classified_types)
        
    def test_validate_model2_svm(self):
        self.test_load_nomenclature2()
        self.usc.load_model('./test/mri_types-test_svm.pkl')
        scans=self.usc.read_scans_csv('./test/all_scans_function.csv')
        classified_types=self.usc.infer_svm(scans)
        self.prediction_accuracy(scans,classified_types)        
                            
    def test_validate_model1_nn(self):
        self.test_load_nomenclature1()
        self.usc.load_model_nn('./test/neuro-onc-test.zip')
        scans=self.usc.read_scans_csv('./test/all_scans_hofid.csv')
        classified_types=self.usc.infer_nn(scans)
        self.prediction_accuracy(scans,classified_types)
            
    def test_validate_model2_nn(self):
        self.test_load_nomenclature2()
        self.usc.load_model_nn('./test/mri_types-test.zip')
        scans=self.usc.read_scans_csv('./test/all_scans_function.csv')
        classified_types=self.usc.infer_nn(scans)
        self.prediction_accuracy(scans,classified_types)

    def test_validate_model3_nn(self):
        self.scm.load_from_file('./test/neuro_onc-dcm.json')
        self.usc.load_model_nn('./test/neuro-onc-test3.zip')
        scans=self.usc.read_scans_csv('./test/all_scans_voxelres_hofid.csv')
        classified_types=self.usc.infer_nn(scans)
        self.prediction_accuracy(scans,classified_types)
        
    def test_validate_model3_ext_nn(self):
        self.scm.load_from_file('./test/neuro_onc-dcm.json')
        self.usc.load_model_nn('./test/neuro-onc-test3.zip')
        scans=self.usc.read_scans_csv('./test/all_scans_voxelres_hofid.csv')
        uris=[ s['URI'] for s in scans ]
        labels1,probs1,labels2,probs2,pred_gini_impurity,pred_margin_confidence,series_descriptions=self.usc.infer_nn_ext(scans)    
        d={'URIs':uris,'labels1':labels1,'probs1':probs1,'labels2':labels2,'probs2':probs2,'series_descriptions':series_descriptions, 'pred_gini_impurity':pred_gini_impurity,'pred_margin_confidence':pred_margin_confidence}    
        with open('./test/all_scans_voxelres_classification_output.csv',mode='w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=d.keys())
            w.writeheader()
            for row in zip(*d.values()): w.writerow(dict(zip(d.keys(),row)))
      
        classified_types=self.usc.infer_nn(scans)
        self.prediction_accuracy(scans,labels1)

    
    def test_infer_model2_svm(self):
        self.usc.load_model('./test/mri_types-test_svm.pkl')
        scans=self.usc.read_scans_csv('./test/all_scans.csv')
        classified_scans=self.usc.predict_classifier(scans)
        self.usc.write_scans_csv(classified_scans,'./test/all_scans-mri_types_predicted_svm.csv')
        
def parse_args():
    parser = argparse.ArgumentParser(description='Classify a list of DICOM files using a pre-trained perceptron model.')
    #parser.add_argument('dicom_files', type=str, nargs='+', help='List of paths to DICOM files to be classified.')
    parser.add_argument('--file_list', type=str, help='file with input DICOM file list', required=True)
    parser.add_argument('--model_file', type=str, help='trained model file (zip)',required=True)
    parser.add_argument('--nomenclature_file', type=str, help='nomenclature file',required=True)
    parser.add_argument('--path_type', type=str, help='XNAT path type (scan,experiment,project)',required=True)
    parser.add_argument('--tag_out', type=str, action='append', help='optional DICOM tag (string name, can be repeated) to output in csv',required=False)
    
    return parser.parse_args()

def parse_paths(paths, path_type):
    '''
    extract scan and experiment ID's from file paths, to put in the output csv.
    '''
    experiments,scans=[],[]
    for path in paths:
        experiment_id = 'NA'
        scan_id = 'NA'
        
        if path_type == 'project':
            match = re.match(r'.*/([^/]+)/([^/]+)/SCANS/([^/]+)/DICOM/([^/]+)', path)
            if match:
                experiment_id = match.group(2)
                scan_id = match.group(3)
        elif path_type == 'experiment':
            match = re.match(r'.*/SCANS/([^/]+)/DICOM/([^/]+)', path)
            if match:
                scan_id = match.group(1)
                
        experiments+=[experiment_id]
        scans+=[scan_id]
        
    return experiments,scans

            
def main():
    args = parse_args()
    #dicom_files = args.dicom_files
    dicom_files=[]
    file_list=args.file_list
    model_file = args.model_file
    nomenclature_file=args.nomenclature_file
    tags_out=args.tag_out
    if tags_out is None: tags_out=[]
    path_type=args.path_type
    
    # Verify that the specified DICOM files exist
    if not os.path.exists(file_list):    
        print("Error: input file list {} does not exist".format(file_list))
        sys.exit(1)
    
    with open(file_list, 'r') as file:
        for line in file:
            dicom_files.append(line.strip())

    for file in dicom_files:
        if not os.path.exists(file):
            print(f"Error: Specified DICOM file does not exist: {file}", file=sys.stderr)
            sys.exit(1)
            
    print('classifying {} files.'.format(len(dicom_files)))
    scm=ScanClassificationModel()
    usc=UniversalScanClassifier(scm)
    if not scm.load_from_file(nomenclature_file): 
        printf(f"Error: cannot read nomenclature file {file}", file=nomenclature_file)
        sys.exit(1)
    if not usc.load_model_nn(model_file):
        printf(f"Error: cannot read model file {file}",file=model_file)
        sys.exit(1)
        
    scans=usc.scans_from_files(dicom_files)
    print("number of input scans:",len(scans))
    labels1,probs1,labels2,probs2,pred_gini_impurity,pred_margin_confidence,series_descriptions=usc.infer_nn_ext(scans)
    print("lengths of output arrays:",len(labels1),len(probs1),len(labels2),len(probs2),len(pred_gini_impurity),len(pred_margin_confidence),len(series_descriptions))   
    d={'files':dicom_files,'labels1':labels1,'probs1':probs1,'labels2':labels2,'probs2':probs2,'series_descriptions':series_descriptions, 'pred_gini_impurity':pred_gini_impurity,'pred_margin_confidence':pred_margin_confidence}

    if len(scans) != len(labels1): 
        print("ERROR: number of input files doesn't match the number of labels, output invalid!")
        return 1	
        
    #add experiment and scan columns.    
    d['experiment'],d['scan']=parse_paths(dicom_files,path_type)
    
    for tag in tags_out:
        d[tag]=[ s[tag] if tag in s.keys() else 'NA' for s in scans ]    
    
    with open('classification_output.csv',mode='w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=d.keys())
        w.writeheader()
        for row in zip(*d.values()): w.writerow(dict(zip(d.keys(),row)))
        
    
if __name__ == '__main__':
    main()
