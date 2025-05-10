# E115_SMART helm set up
The project uses Helm deployment tool to set up deployment to Kubernetes.

## First Time Deployment

### Transfer secret Google OAuth

This would ideally be done with the SOPS plugin for helm, but was done mannually for simplicity.

```
kubectl create secret generic oauth-secret --from-file=client_secrets.json=client_secrets2.json
```

### Transfer data to api pod

This is necessary because the datapipeline cannot run in the cluster itself without a GPU.
As a workaround the datapipeline was run locally and the database was populated by hand.

**Get data from docker**
```
docker exec -it postgres /bin/bash
pg_dump -U postgres -d smart -f /var/lib/postgresql/data/dump.sql
sudo cp ../persistent-folder/postgres/dump.sql ../E115_SMART/
```

**Push data to postgres pod**

1. Bring down the api pod
2. Exec into postgres pod
3. Delete the existing database
4. Create database
5. Copy the file
6. Bring back the api pod

```
kubectl scale deployment smart-api --replicas=0

kubectl cp dump.sql smart-postgres-7b58dd9b9f-lhkv9:/var/lib/postgresql/data
kubectl exec -it smart-postgres-7b58dd9b9f-lhkv9 -- /bin/bash

psql -U postgres
drop database smart;
create database smart;
exit

psql -U postgres -d smart -f /var/lib/postgresql/data/dump.sql

kubectl scale deployment smart-api --replicas=1
```

## Deploy Code Changes (CI/CD)

1. Make Changes in local repository
2. Git Commit + Git Push
3. Check GitHub Actions for success - deployment is automatic

```
git add .
git commit -m "<message>"
git push
```

## Evidence of Auto scaling
Lowered the limit of the api pod in the kubernetes deployment (via helm) to 1Gi to evidence autoscaling.

1. Pre auto scaling image
```
watch kubectl top pods
```

![Pre scale](/images/before_auto_scale.png)

2. Keep clicking on below until it shows autocals
```
https://smart.ghundal.com/eat-mem
```

3. Evidence of autoscaling

![Post scale](/images/after_auto_scale.png)

## Organization
```
├── .github/workflows
|      |── ci_cd.yaml
|      └── pre-commit.yaml
|
└── helm
|   ├── templates
|   |       |── _helpers.tpl
|   |       |── configmap.yaml
|   |       |── deployment.yaml
|   |       |── hpa.yaml
|   |       |── ingress.yaml
|   |       |── NOTES.txt
|   |       |── pvc.yaml
|   |       |── secret.yaml
|   |       |── service.yaml
|   |       |── serviceaccount.yaml
|   |       └── tls-secret.yaml
│   ├── .helmignore
│   ├── Chart.yaml
│   ├── README.md
│   ├── value.yaml
├── .pre-commit-config.yaml
└── README.md
```

## Steps to create deployment scripts

**Install helm**
```
snap install helm
```

**Generate template**
```
helm create smart
```
## Troubleshooting

### Rendering the chart
```
helm template smart ./helm
```

### Check running pods
```
kubectl get pods
```

### Check Logs
```
kubectl logs <pod name>
```

### Access postgres database in Kubernetes with shell
```
kubectl exec -it <postgres pod name>  -- /bin/bash
root: psql -U postgres
\c smart
\d
```
### Get deployments
```
kubectl get deployments
```

### Manually Scale up or down (check with get pods or deplyments)
```
kubectl scale deployment <artifact> --replicas=<number>

kubectl scale deployment smart-frontend --replicas=0
kubectl scale deployment smart-frontend --replicas=3
```

### Describe (Check status)
```
kubectl describe pod <pod name>
```
