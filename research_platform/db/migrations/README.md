# Database migrations

Migrations target the existing MySQL 8.0+ service.

## First migration

Run `001_initial.sql` once against the platform database after verifying the
connection and database name. Do not run it against a production trading
schema or any database containing unrelated application data.

The migration stores metadata only. It does not move or modify existing
`quotes/`, `output/`, frozen strategies, or model files.

A future migration runner must:

1. acquire a migration lock;
2. read `platform_schema_version`;
3. apply each unapplied file in lexical order;
4. record the version only after success;
5. fail closed on partial application.
