ALTER ROLE rag_user SET client_encoding TO 'utf8';
ALTER ROLE rag_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE rag_user SET timezone TO 'UTC';

GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;
