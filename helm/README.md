# E115_SMART helm set up
The project uses Helm deployment tool to set up deployment to Kubernetes.

## Prerequisites

**Install helm**
```
snap install helm
```

**Generate template**
```
helm create smart
```

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
kubectl exec -it < postgres pod name>  -- /bin/bash
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
kubectl scale deployment <artifact> --replicas= <number>

kubectl scale deployment smart-frontend --replicas=0
kubectl scale deployment smart-frontend --replicas=3
```

### Describe (Check status)
```
kubectl describe pod <pod name>
```



### Deployment

**Transfer secret Google OAuth**
```
kubectl create secret generic oauth-secret --from-file=client_secrets.json=client_secrets2.json
```

**Transfer data to api pod**
Get data from docker
```
docker exec -it postgres /bin/bash
pg_dump -U postgres -d smart -f /var/lib/postgresql/data/dump.sql
sudo cp ../persistent-folder/postgres/dump.sql ../E115_SMART/
```

Push data to postgres pod
```
kubectl cp dump.sql smart-postgres-65d584dfd4-lcx57:/var/lib/postgresql/data
kubectl exec -it smart-postgres-65d584dfd4-lcx57 -- /bin/bash
psql -U postgres -d smart -f /var/lib/postgresql/data/dump.sql
```
