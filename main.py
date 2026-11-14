import json, argparse
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def train(out='outputs'):
    d=Path(out);d.mkdir(parents=True,exist_ok=True)
    X,y=load_breast_cancer(return_X_y=True)
    assert np.isfinite(X).all() and len(np.unique(y))==2
    a,b,c,e=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42)).fit(a,c)
    pred=model.predict(b); prob=model.predict_proba(b)[:,1]
    metrics={'accuracy':float(accuracy_score(e,pred)),'f1':float(f1_score(e,pred)),'auroc':float(roc_auc_score(e,prob)),'n_train':len(a),'n_test':len(b)}
    joblib.dump(model,d/'model.joblib');np.save(d/'reference_mean.npy',a.mean(0));np.save(d/'reference_std.npy',a.std(0)+1e-8)
    (d/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics

def drift(batch,out='outputs'):
    d=Path(out);mu=np.load(d/'reference_mean.npy');sd=np.load(d/'reference_std.npy');z=np.abs((np.asarray(batch).mean(axis=0)-mu)/sd)
    return {'mean_standardized_shift':float(z.mean()),'features_over_2sd':int((z>2).sum())}

def predict(rows,out='outputs'):
    model=joblib.load(Path(out)/'model.joblib');return model.predict_proba(np.asarray(rows))[:,1].tolist()

def create_app(out='outputs'):
    from fastapi import FastAPI
    from pydantic import BaseModel
    app=FastAPI(title='MLOps360 Demo API')
    class Input(BaseModel): features:list[list[float]]
    @app.get('/health')
    def health():return {'status':'ok'}
    @app.post('/predict')
    def inference(x:Input):return {'probabilities':predict(x.features,out)}
    return app

if __name__=='__main__':print(json.dumps(train(),indent=2))
