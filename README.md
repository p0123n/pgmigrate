# PGmigrate

PostgreSQL migrations made easy

## Overview

Actively maintained fork of the Yandex PGmigrate tool with several improvements.

Goals of this fork:

* **Make this tool more flexible**
* **Support main features of the original version** 

## Install

```
pip install -e git+https://github.com/p0123n/pgmigrate.git#egg=yandex-pgmigrate --upgrade
```

## Differents from the original PGmigrate

* **Migration creation :)**
In this fork you can easily make new migrations with command:
```
pgmigrate create -m "my new migration"
```

* **Multiple configuration support**
When you need to manage several migration profiles (e.g. dev, testing, local), you need to use `-c` option to pass configuration. 
It's not frendly in all cases, so you can use following syntax:
```
pgmigrate --cfg local --base_dir migrations/local info
```
The `--cfg` option - name of the configuration file. 
In this case config file will be `migrations_local.yml`

You can use this feature in following cases:
- Split migration to the `data` and `structure` migrations
- When you want keep configuraion files for several environments

* **Old Python versions (<=3.6) not supported**
Please use original tool if you need python old versions support.

## License

Distributed under the PostgreSQL license. See [LICENSE](LICENSE) for more
information.
