# e115_SMART
Repo for project for E115

## Docker login
```
docker login
``` 

## Access the database
### Ubuntu
```
sudo apt update && sudo apt install -y postgresql-client
psql -U postgres -h localhost
```
### OS independent
```
docker exec -it postgres /bin/bash
psql -U postgres
```
