import argparse, warnings, pickle, re, numpy as np
from sklearn.feature_extraction.text import CountVectorizer

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import tensorflow as tf

'''
Author: Mikhail Milchenko, mmilchenko@wustl.edu
Copyright (c) 2021, Computational Imaging Lab, School of Medicine, Washington University in Saint Louis

Redistribution and use in source and binary forms, for any purpose, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

class HOF_Classifier:
    def __init__(self):
        self.classifier=[]
        self.vectorizer=[]
        self._class_vectorizer=None
        #important: must be in aphpabetical order for vectorizer to work correctly.
        self._classes=['CBF','CBV','DSC','DWI','FA','MD','MPRAGE','MTT','OT','PBP','SWI','T1hi','T1lo','T2FLAIR','T2hi','T2lo','TRACEW','TTP']
        #self._scan_list=[]
    def load_json(self, json_file):
        with open(json_file, 'r') as fp:
            out_dict=json.loads(fp.read())
        return out_dict    
    def save_json(self, var, file):
        with open(file,'w') as fp:
            json.dump(var, fp) 
    '''
    Assign HOF ID's to scans using associative table look-up.
    '''
    def assign_hofids_slist(self,scans):
        for s in scans:
            descr=re.sub(' ','',s['series_description'])
            cmd="slist qd "+"\"" + descr + "\""
            try:
                hof_id=os.popen(cmd).read().split()[1]
            except:
                hof_id=""
            #print(hof_id)
            s['hof_id']=hof_id
            #out.value="{}/{}".format(s['series_description'],hof_id)
        return scans
    
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
    
    def prepare_descs(self,scans):
        #descs are 'sentences' that contain series description and log-compressed number of frames.
        descs=[]
        for s in scans:
            desc=(re.sub('[^0-9a-zA-Z ]+',' ',s['series_description'])).split()
            #compressed representation of the number of frames.
            try:
                frames='frames{}'.format(str(int(np.around(np.log(1.0+float(s['frames']))*3.0))))
            except:
                frames='frames0'
            desc.append(frames)
            descs.append(' '.join([s for s in desc if ((not s.isdigit()) and (len(s)>1)) ]))
        return descs
        
    def gen_bow_vectors(self,scans):
        if not self.vectorizer: return []
        descs=self.prepare_descs(scans)
        return self.vectorizer.transform(descs).toarray()    
    
    def train_nn(self,X,y,test_split,epochs=10,batch_size=10):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_split,random_state=1000)
        input_dim=X_train.shape[1]
        print('input_dim:',input_dim)
        model = Sequential()
        model.add(layers.Dense(36,input_dim=input_dim,activation='relu'))
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
    
    def _merge_hofids(self,scans,hofids):
        for s in scans:
            descr=re.sub(' ','',s['series_description'])
            cmd="slist qd "+"\"" + descr + "\""
            try:
                hof_id=os.popen(cmd).read().split()[1]
            except:
                hof_id=""
            #print(hof_id)
            s['hof_id']=hof_id
            out.value="{}/{}".format(s['series_description'],hof_id)
        
    def _predict_classifier(self,X):
        if not self.classifier: return []
        return self.classifier.predict(X)
        
    def predict_classifier(self, scans):
        vectorized_descs=self.gen_bow_vectors(scans)
        labels=self._predict_classifier(vectorized_descs)
        for i,s in enumerate(scans):
            s['hof_id']=labels[i]
        return scans
    
    def is_valid_model(self):
        return (self.vectorizer and self.classifier)    
        
    def save_model_nn(self,rt):
        pickle.dump(self.vectorizer,open(rt+'.vec','wb'))
        self.classifier.save(rt+'.hd5')
        
    def load_model_nn(self,rt):
        self.vectorizer=pickle.load(open(rt+'.vec','rb'))
        self.classifier=tf.keras.models.load_model(rt+'.hd5')
    
    def save_model(self, file):
        pickle.dump([self.vectorizer,self.classifier],open(file,'wb'))
                    
    def load_model(self, file):
        self.vectorizer,self.classifier=pickle.load(open(file,'rb'))    

def classify_scans(frames:list,descrs:list,hc:HOF_Classifier):
    '''
    input list of numbers of frames and series descriptions
    return comma separated list of HOF IDs
    '''
    
    scans=[]
    for f,d in zip(frames,descrs):
        scans.append(dict(series_description=d,frames=f))
    hof_ids=hc.infer_nn(scans)
    return ','.join(hof_ids)

def get_parser():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Assign a HOF type based on scan metadata to multiple scans')

    # Positional arguments.
    parser.add_argument("series_descriptions", help="DICOM series descriptions, comma separated")
    parser.add_argument("numbers_of_frames", help="Numbers of frames,comma separated")
    parser.add_argument("model", help=" model file root")
    
    return parser.parse_args()
    
    
if __name__ == "__main__":
    p = get_parser()
    
    
    frames=p.numbers_of_frames.split(',')
    descrs=p.series_descriptions.split(',')
    hc=HOF_Classifier()
    hc.load_model_nn(p.model)
        
    print(classify_scans(frames,descrs,hc))
    
    # Options
    #parser.add_argument("--out_struct", metavar="<string>",type=str,default=None, 
    #                    help="Output structural image root [output_nifti_rtss+struct.nii]")
    #parser.add_argument("--exclude_labels", metavar="<string>",type=str,default=None,
    #                    help="Comma separated list of ROI labels to exclude, case insensitive [None]")
    #parser.add_argument("--separate_masks", action="store_true", default=False, help="write each ROI mask in a separate file [False]")
    