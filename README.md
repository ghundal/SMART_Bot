# S.M.A.R.T. — Secure ‘doc’ Management And Retrieval Technology

Organizations increasingly rely on internal digital repositories—notes, policies, records—but conventional keyword-based search fails to deliver contextual understanding. SMART addresses this gap by providing an intelligent, secure, and attribution-aware retrieval system powered by LLMs and hybrid search while prioritizing data privacy and access control.

App Link: https://smart.ghundal.com/

Presentation Link: [Presentation](./reports/MS5_SMART_Final.pptx)

SMART delivers:

- **Secure Document Storage:** Centralized, permissioned content repository (class notes, quizzes).
- **Semantic Search & Ranking:** Vector embeddings via LLM with vector and BM25 hybrid search.
- **Guardrails:** LLM powdered guardrails to avoid jailbreaking and inappropriate content.
- **LLM-Powered Summarization:** Relevant results reranked and structured via LLMs, supporting multilingual text.
- **Frontend UI:** Chatbot interface, authenticated via Google OAuth with SSL encryption.
  Security + Audit Trail: Logs at every access and retrieval point.

## **Organization**

```
E115_SMART/
├── .github
├── ansible
├── helm
├── images
├── linter
├── reports
├── sql
├── src
├── tests
|
└── README.md
```

## **Overview**

Please reference the [Design Document](./reports/Design_Document.pdf) for the full tech stack.

## **SMART Artifacts**

![SMART Artifacts](./images/artifacts.png)

## **Technical Components**

1. Data Pipeline

![Data Pipeline Component](./images/DataPipeline.png)
The documents, access, and metadata are stored in the Google Cloud Storage bucket. The application pulls the data from the GCS bucket, performs validations to ensure that data is complete and correct, semantically chunks the data, and then embeds the final chunks. The chunks along with metadata and access are stored in postgres database.

2. Application Programming Interface (API)

![API Component](./images/API.png)

The user writes a query in the frontend. The query is first checked for safety by Llama Guard in the original language. If the query violates the guidelines, SMART returns the answer that the query violates the guidelines and stops the flow. If the query is safe, a voting based system detects the language of the query. If the query is in English, it is directly embedded. If the query is non-English, it is translated to English and then again checked for safety by Llama Guard. Once safe, it proceeds to be embedded.

The search function prefilters the chunks based on the user access, then performs a keyword and vector search to return the most relevant documents and their metadata. That information is sent in parallel to the ranker and one of the chosen LLM models (Llama3 or Gemma3) which returns an answer and the three most relevant chunks. The answer is translated back into the original language if non-English and then returned to the user in the frontend.

3. Deployment

![Deployment Component](./images/deployment.png)

SMART’s CI/CD (Continuous Integration/Continuous Deployment) pipeline is triggered whenever code changes are committed to GitHub. The GitHub Actions workflow runs a comprehensive suite of pre-commit checks, including linter, formatter, unit tests, and integration tests. Once all tests pass, the deployment process is initiated automatically.

Ansible scripts verify the Kubernetes cluster's status in Google Kubernetes Engine (GKE), ensuring it is ready for deployment. Helm scripts then manage the deployment within the GKE cluster, streamlining the rollout of new updates.

## **Solution Architecture**

![Solution Architecture](./images/solutionArchitecture.png)

## **Technical Architecture**

![Technical Architecture](./images/technicalArchitecture.png)

## **Local Deployment**

Refer to individual containers for local deployment steps:

1. [Data Pipeline](./src/datapipeline/README.md)
2. [API](./src/api/README.md)
3. [Frontend](./src/frontend/README.md)

## **Tests**

The tests consist of:

- Linter and formatter
- Unit test for Python scripts
- Unit test for Frontend
- Validation test for Frontend
- Integration tests

To run the linter locally refer:
[Linter Readme](/linter/README.md)

To run the test locally refer:
[Test Readme](/tests/README.md)

## **Production Deployment**

Please reference [deployment README.md](/helm/README.md) for the steps and the evidence of autoscaling.

## **Evidence of Deployment**

Below is the evidence of successfull test run and CI/CD deployment.
For the latest logs, you can refer to the GitHube Actions.

![Evidence](/images/Evidence_Deployment.png)

## **Usage details**

Please refer to [SMART User Guide](/reports/SMART_UserGuide.pdf) for details on screenshots and usage.

## **Limitations and improvements**

1. **No GPU**
   Despite our best efforts, we were not able to get a quota for a GPU on GCP. Hence the production version is much slower (response time: 15-25 min) than local (response time: 10-15 seconds). Ideally, we would like to run the pipeline on kubernetes provided we have a GPU.

2. **Security Enhancements**
   Deploy a service mesh like Istio, Linkerd, or Consul to handle mutual TLS (mTLS) between pods. The service mesh automatically encrypts traffic, manages certificates, and enforces policies. Utilize Kubernetes secrets or a certificate manager (e.g., cert-manager) to generate and distribute TLS certificates to pods. Enable mTLS to verify the identity of communicating pods.

3. **Architecture Enhancements**
   Remove the dependency on the third party models and distill our own language models. Additionally, implement Reinforcement Learning from Human Feedback (RLHF) and confidence algorithm for LLM's answer.

4. **Frontend Enhancements**
   Include an admin level frontend to load the documents, thus removing dependency on Google Cloud Storage. This requires a GPU.
