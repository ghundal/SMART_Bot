# Frontend

This module is desined to start the fontend locally.

## **Prerequisites**

Ensure that datapipeline and api are set up and running.

## Setup Google Credentials Account\*\*

Go to [Google console](https://console.developers.google.com/) and ensure that you have added the below in the credentials for Google authentications.

http://localhost:9000/auth/auth

https://smart.ghundal.com/auth/auth

Extract the client secrets and add to the below folder.

```
|-E115_SMART
|-secrets
  |-client_secrets.json
```

## Organization

```
├── Readme.md
├── sql
│   └── init.sql
|
└── src
    ├── api
    ├── datapipeline
    ├── frontend
    │   ├── app
    |   |       |── about
    |   |       |  └── page.jsx
    |   |       |── chat
    |   |       |  └── page.jsx
    |   |       |── login
    |   |       |  └── page.jsx
    |   |       |── reports
    |   |       |      └── page.jsx
    |   |       |── global.css
    |   |       |── layout.jsx
    |   |       |── not_found.jsx
    |   |       |── page.jsx
    |   |       └── page.module.css
    │   ├── components
    |   |       |── about
    |   |       |  └── About.module.css
    |   |       |── auth
    |   |       |  |── LoginButton.jsx
    |   |       |  └── ProtectedRoute.jsx
    |   |       |── chat
    |   |       |  └── chat.module.css
    |   |       |── layout
    |   |       |      |── Footer.jsx
    |   |       |      |── Footer.module.css
    |   |       |      |── Header.jsx
    |   |       |      └── Header.module.css
    |   |       └── reports
    |   |             |── exportUtils.js
    |   |             └── reports.module.css
    |   ├── context
    |   |       └── AuthContext.jsx
    |   ├── public
    |   |       └── logo.png
    |   ├── services
    |   |       |── DataService.js
    |   |       └── ReportService.js
    │   ├── .env.local
    │   ├── Dockerfile
    │   ├── jsconfig.json
    │   ├── package-lock.json
    │   ├── package.json
    │   ├── README.md
    │   └── tailwind.config.js
    ├── docker-compose.yml
    ├── docker-shell.sh
    └── Dockerfile.postgres
```

## **Running the frontend**

Execute the below command in /src

```bash
sh docker-shell.sh frontend
```
