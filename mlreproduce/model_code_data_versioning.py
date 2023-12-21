#!/usr/bin/env python
# coding: utf-8


class DataCodeModelVersioning:
    
    def __init__(self):
        from minio import Minio
        from minio.error import S3Error
        import os
        import mlflow
        self.mlflow_url="https://uat-dasec-mlflow.dastc.stee.com/"
        self.bucket="versioned-code-data"
        self.minio_client = Minio(
        "dasec-lakehouse-minio.dastc.stee.com:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        )
        #MLFLOW and MinIO Setup
        os.environ['MLFLOW_TRACKING_URI']='https://uat-dasec-mlflow.dastc.stee.com/'
        os.environ['AWS_ACCESS_KEY_ID']='minioadmin'
        os.environ['AWS_SECRET_ACCESS_KEY']='minioadmin'
        os.environ['MLFLOW_S3_ENDPOINT_URL']='https://dasec-lakehouse-minio.dastc.stee.com:9000'
                     
        
    
    def set_data_code_model_versioning(self,project_name,model_name,model,data_file,code_file,params,metrics):
        
        """creates versions of data,code and model""" 
        # Upload data.
        result_data = self.minio_client.fput_object(
                self.bucket, f'{project_name}/{model_name}/{data_file}', f'{data_file}',)
        result_code = self.minio_client.fput_object(
                self.bucket, f'{project_name}/{model_name}/{code_file}', f'{code_file}',)
        import mlflow
        mlflow.set_tracking_uri(self.mlflow_url)
        experiment_name=project_name
        registered_model_name=model_name
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.log_params(params)
            mlflow.set_tags({
                'data_version':[result_data.object_name,result_data.version_id],
                'code_version':[result_code.object_name,result_code.version_id]
            })

            mlflow.sklearn.log_model(
                sk_model=model, artifact_path=registered_model_name,registered_model_name=registered_model_name
            )
        print("data, code and model are versioned sucessfully!!! for more details, visit MLFlow page({})".format(self.mlflow_url))
    
    def get_versioned_data(self,versioned_data=[]):
        import os
        # Download data of an object of version-ID.
        self.minio_client.fget_object(
        self.bucket, f"{versioned_data[0]}",f"{versioned_data[0].split('/')[-1]}",version_id=f"{versioned_data[1]}",
        )
        
        return os.path.abspath(f"{versioned_data[0].split('/')[-1]}")
    def get_versioned_code(self,versioned_code=[]):
        import os
        # Download data of an object of version-ID.
        self.minio_client.fget_object(
        self.bucket, f"{versioned_code[0]}",f"{versioned_code[0].split('/')[-1]}",version_id=f"{versioned_code[1]}",
        )
        
        return os.path.abspath(f"{versioned_code[0].split('/')[-1]}")
    
    def get_versioned_model(self,model_name,model_version):
        import mlflow
        mlflow.set_tracking_uri("https://uat-dasec-mlflow.dastc.stee.com/")
        
        return mlflow.pyfunc.load_model(
               model_uri=f"models:/{model_name}/{model_version}"
        )