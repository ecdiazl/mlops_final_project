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
            packages_to_install=["google-cloud-bigquery==3.10.0"],
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
    
if __name__ == '__main__':
    
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="tab_classif_pipeline.json"
    )
    print('Pipeline compilado exitosamente')
