import time
from datetime import datetime
from typing import NamedTuple

import kfp
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component, pipeline)
from kfp.v2.google.client import AIPlatformClient

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip

PROJECT_ID = 'mlops-final-project-412223'
DATASET_ID = "data_mlops_fp"  
TABLE_ID = "tb_mlops_fp"  

@component(
            packages_to_install=["google-cloud-bigquery[pandas]==3.10.0"],
)
def read_bigquery_table(
    project_id: str, 
    dataset_id: str, 
    table_id: str,
    dataset: Output[Dataset],
):
    """
    Lee datos de una tabla de BigQuery y devuelve algún resultado como ejemplo.
    
    Args:
    project_id: ID del proyecto en GCP.
    dataset_id: ID del conjunto de datos en BigQuery.
    table_id: ID de la tabla en el conjunto de datos.
    
    Returns:
    Una cadena que representa una parte de los datos leídos, por ejemplo.
    """
    from google.cloud import bigquery

    # Crea un cliente de BigQuery.
    client = bigquery.Client(project=project_id)

    # Define la consulta para seleccionar todos los datos de la tabla.
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` "
    
    # configuramos la consulta
    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=query, job_config=job_config)

    # Convierte los resultados en un dataframe de pandas y
    # escribe los resultados en un archivo CSV.
    df = query_job.result().to_dataframe()
    df.to_csv(dataset.path, index=False)

@component(
    packages_to_install=[
        "pandas==1.3.5",
        "scikit-learn==1.0.2",
    ],
)
def data_preprocessing(
    dataset: Input[Dataset],
    scaled_dataset: Output[Dataset],
):
    """Preprocesses tabular data for training."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # lee los datos de entrada
    df = pd.read_csv(dataset.path)

    # separamos el target del resto de las variables
    X, y = df.iloc[:,1:-1], df[['target']]
    columns_x = X.columns

    # escalamos las variables
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=columns_x)

    # concatenamos las variables escaladas con el target
    df_scaled = pd.concat([X, y], axis=1)

    # guardamos el archivo como CSV
    df_scaled.to_csv(scaled_dataset.path, index=False)


@component(
    packages_to_install=[
        "pandas==1.3.5",
        "scikit-learn==1.0.2",
        "joblib==1.1.0",
    ],
)
def model_training(
    scaled_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
):
    """Trains a model on tabular data."""
    import pandas as pd
    import joblib
    import os
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import (accuracy_score, precision_recall_curve,
                                 roc_auc_score)
    from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                         train_test_split)
    from sklearn.preprocessing import LabelEncoder

    # read the data
    df = pd.read_csv(scaled_dataset.path)

    # split the data into X and y
    X, y = df.iloc[:, :-1], df[['target']]

    # Assuming X and y are predefined
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    params = {
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "penalty": ["l2", "l1", "elasticnet"],
        "loss": ["hinge", "log", "modified_huber"],
        "max_iter": [1000, 2000, 3000]
    }

    sgd_model = SGDClassifier(random_state=42)

    folds = 3
    param_comb = 10

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        sgd_model,
        param_distributions=params,
        n_iter=param_comb,
        scoring="precision",
        n_jobs=4,
        cv=skf.split(X_train, y_train),
        verbose=4,
        random_state=42,
    )

    random_search.fit(X_train, y_train)
    sgd_model_best = random_search.best_estimator_
    predictions = sgd_model_best.predict(X_test)
    score = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    _ = precision_recall_curve(y_test, predictions)

    metrics.log_metric("accuracy", (score * 100.0))
    #metrics.log_metric("framework", "xgboost")
    metrics.log_metric("dataset_size", len(df))
    metrics.log_metric("AUC", auc)

    # Export the model to a file
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(sgd_model_best, os.path.join(model.path, "model.joblib"))


@dsl.pipeline(
    name="mlops-fp-pipeline",
)
def pipeline():
    """A demo pipeline."""

    read_bigquery_table_task = read_bigquery_table(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
    )

    data_preprocessing_task = (
        data_preprocessing(
            dataset=read_bigquery_table_task.outputs["dataset"],
        )
        .after(read_bigquery_table_task)
        .set_caching_options(False)
    )

    model_training_task = (
        model_training(
            scaled_dataset=data_preprocessing_task.outputs["scaled_dataset"],
        )
        .after(data_preprocessing_task)
        .set_caching_options(False)
    )

# Deploy the model to Vertex AI
if __name__ == '__main__':
    
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="tab_classif_pipeline.json"
    )
    print('Pipeline compilado exitosamente')
