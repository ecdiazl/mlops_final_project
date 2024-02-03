from datetime import datetime
from google.cloud import aiplatform


if __name__ == '__main__':
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    job = aiplatform.PipelineJob(
        display_name="mlops-fp-sgd",
        template_path="sgd_classif_pipeline.json",
        job_id="mlops-fp-sgd-{0}".format(TIMESTAMP),
        enable_caching=True
    )

    job.submit()
    
    print('Pipeline successfully submitted')