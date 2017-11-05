#!/usr/bin/env python
"""
PGmigrate - PostgreSQL migrations made easy
"""
# -*- coding: utf-8 -*-
#
#    Copyright (c) 2016-2017 Yandex LLC <https://github.com/yandex>
#    Copyright (c) 2016-2017 Other contributors as noted in the AUTHORS file.
#
#    Permission to use, copy, modify, and distribute this software and its
#    documentation for any purpose, without fee, and without a written
#    agreement is hereby granted, provided that the above copyright notice
#    and this paragraph and the following two paragraphs appear in all copies.
#
#    IN NO EVENT SHALL YANDEX LLC BE LIABLE TO ANY PARTY FOR DIRECT,
#    INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST
#    PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
#    EVEN IF YANDEX LLC HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#    YANDEX SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
#    BASIS, AND YANDEX LLC HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE,
#    SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import codecs
import json
import logging
import os
import re
import sys
from builtins import str as text
from collections import OrderedDict, namedtuple
from datetime import datetime

import psycopg2
import sqlparse
import yaml
from psycopg2.extras import LoggingConnection

LOG = logging.getLogger(__name__)
MIGRATION_CONFIG = 'migrations.yml'
MIGRATION_FOLDER = 'migrations'
MIGRATION_DEFAULT_ENCODING = u'/* pgmigrate-encoding: utf-8 */'
MIGRATIONS_TABLE_NAME = 'schema_version'


class MigrateError(RuntimeError):
    """
    Common migration error class
    """
    pass


class MalformedStatement(MigrateError):
    """
    Incorrect statement exception
    """
    pass


class MalformedMigration(MigrateError):
    """
    Incorrect migration exception
    """
    pass


class MalformedSchema(MigrateError):
    """
    Incorrect schema exception
    """
    pass


class ConfigParseError(MigrateError):
    """
    Incorrect config or cmd args exception
    """
    pass


class BaselineError(MigrateError):
    """
    Baseline error class
    """
    pass


REF_COLUMNS = ['version', 'description', 'type',
               'installed_by', 'installed_on']


def _create_connection(conn_string):
    conn = psycopg2.connect(conn_string, connection_factory=LoggingConnection)
    conn.initialize(LOG)

    return conn


def _init_cursor(conn, session):
    """
    Get cursor initialized with session commands
    """
    cursor = conn.cursor()
    for query in session:
        cursor.execute(query)
        LOG.info(cursor.statusmessage)

    return cursor


def _is_initialized(cfg):
    """
    Check that database is initialized
    """
    query = cfg.cursor.mogrify('SELECT EXISTS(SELECT 1 FROM '
                               'information_schema.tables '
                               'WHERE table_schema = %s '
                               'AND table_name = %s);',
                               ('public', cfg.table_name))
    cfg.cursor.execute(query)
    table_exists = cfg.cursor.fetchone()[0]

    if not table_exists:
        return False

    cfg.cursor.execute(f'SELECT * from public.{cfg.table_name} limit 1;')

    colnames = [desc[0] for desc in cfg.cursor.description]

    if colnames != REF_COLUMNS:
        raise MalformedSchema(f'Table {cfg.table_name} has unexpected '
                              f'structure: %s' % '|'.join(colnames))

    return True


MIGRATION_FILE_RE = re.compile(
    r'V(?P<version>\d+)__(?P<description>.+)\.sql$'
)

MigrationInfo = namedtuple('MigrationInfo', ('meta', 'filePath'))

Callbacks = namedtuple('Callbacks', ('beforeAll', 'beforeEach',
                                     'afterEach', 'afterAll'))

Config = namedtuple('Config', ('target', 'baseline', 'cursor', 'dryrun',
                               'callbacks', 'user', 'base_dir', 'conn',
                               'session', 'conn_instance', 'table_name',
                               'message'))

CONFIG_IGNORE = ['cursor', 'conn_instance']


def _get_migrations_info_from_dir(base_dir):
    """
    Get all migrations from base dir
    """
    path = os.path.join(base_dir, MIGRATION_FOLDER)
    migrations = {}
    if os.path.exists(path) and os.path.isdir(path):
        for fname in os.listdir(path):
            file_path = os.path.join(path, fname)
            if not os.path.isfile(file_path):
                continue
            match = MIGRATION_FILE_RE.match(fname)
            if match is None:
                continue
            version = int(match.group('version'))
            ret = dict(
                version=version,
                type='auto',
                installed_by=None,
                installed_on=None,
                description=match.group('description').replace('_', ' ')
            )
            ret['transactional'] = 'NONTRANSACTIONAL' not in ret['description']
            migration = MigrationInfo(
                ret,
                file_path
            )
            if version in migrations:
                raise MalformedMigration(
                    'Found migrations with same version: %d ' % version +
                    '\nfirst : %s' % migration.filePath +
                    '\nsecond: %s' % migrations[version].filePath)
            migrations[version] = migration

    return migrations


def _get_migrations_info(cfg, baseline_v):
    """
    Get migrations from baseline to target from base dir
    """
    migrations = {}
    target = cfg.target if cfg.target is not None else float('inf')

    for version, ret in _get_migrations_info_from_dir(cfg.base_dir).items():
        if version > baseline_v and version <= target:
            migrations[version] = ret.meta
        else:
            LOG.info(
                'Ignore migration %r cause baseline: %r or target: %r',
                ret, baseline_v, target
            )
    return migrations


def _get_info(cfg):
    """
    Get migrations info from database and base dir
    """
    ret = {}
    cfg.cursor.execute(f'SELECT ' + ', '.join(REF_COLUMNS) +
                       f' from public.{cfg.table_name};')
    baseline_v = 0

    for i in cfg.cursor.fetchall():
        version = {}
        for j in enumerate(REF_COLUMNS):
            if j[1] == 'installed_on':
                version[j[1]] = i[j[0]].strftime('%F %H:%M:%S')
            else:
                version[j[1]] = i[j[0]]
        version['version'] = int(version['version'])
        transactional = 'NONTRANSACTIONAL' not in version['description']
        version['transactional'] = transactional
        ret[version['version']] = version

        baseline_v = max(cfg.baseline, sorted(ret.keys())[-1])

    migrations_inf = _get_migrations_info(cfg, baseline_v)
    for version in migrations_inf:
        num = migrations_inf[version]['version']
        if num not in ret:
            ret[num] = migrations_inf[version]

    return ret


def _get_database_user(cursor):
    cursor.execute('SELECT CURRENT_USER')
    return cursor.fetchone()[0]


def _get_state(cfg):
    """
    Get info wrapper (able to handle noninitialized database)
    """
    return _get_info(cfg) \
        if _is_initialized(cfg) \
        else _get_migrations_info(cfg, cfg.baseline)


def _set_baseline(cfg):
    """
    Cleanup migrations table and set baseline
    """
    query = cfg.cursor.mogrify(
        f'SELECT EXISTS(SELECT 1 FROM public'
        f'.{MIGRATIONS_TABLE_NAME} WHERE version >= %s::bigint);',
        (cfg.baseline,))
    cfg.cursor.execute(query)
    check_failed = cfg.cursor.fetchone()[0]

    if check_failed:
        raise BaselineError('Unable to baseline, version '
                            '%s already applied' % text(cfg.baseline))

    LOG.info(f'cleaning up table {MIGRATIONS_TABLE_NAME}')
    cfg.cursor.execute(f'DELETE FROM public.{MIGRATIONS_TABLE_NAME};')
    LOG.info(cfg.cursor.statusmessage)

    LOG.info('setting baseline')
    query = cfg.cursor.mogrify(f'INSERT INTO public.{MIGRATIONS_TABLE_NAME} '
                               f'(version, type, description, installed_by) '
                               f'VALUES (%s::bigint, %s, %s, %s);',
                               (text(cfg.baseline), 'manual',
                                'Forced baseline', cfg.user))
    cfg.cursor.execute(query)
    LOG.info(cfg.cursor.statusmessage)


def _init_schema(cursor):
    """
    Create migrations table table
    """
    LOG.info('creating type schema_version_type')
    sql = """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'schema_version_type') THEN
            CREATE TYPE public.schema_version_type AS ENUM (%s, %s);
        END IF;
    END$$;
    """
    query = cursor.mogrify(sql, ('auto', 'manual'))
    cursor.execute(query)
    LOG.info(cursor.statusmessage)
    LOG.info(f'creating table {MIGRATIONS_TABLE_NAME}')
    query = cursor.mogrify(f'CREATE TABLE IF NOT EXISTS public.{MIGRATIONS_TABLE_NAME} ('
                           f'version BIGINT NOT NULL PRIMARY KEY, '
                           f'description TEXT NOT NULL, '
                           f'type public.schema_version_type NOT NULL '
                           f'DEFAULT %s, '
                           f'installed_by TEXT NOT NULL, '
                           f'installed_on TIMESTAMP WITHOUT time ZONE '
                           f'DEFAULT now() NOT NULL);', ('auto',))
    cursor.execute(query)
    LOG.info(cursor.statusmessage)


def _get_statements(path):
    """
    Get statements from file
    """
    with codecs.open(path, encoding='utf-8') as i:
        data = i.read()
    if MIGRATION_DEFAULT_ENCODING not in data:
        try:
            data.encode('ascii')
        except UnicodeError as exc:
            raise MalformedStatement(
                'Non ascii symbols in file: {0}, {1}'.format(
                    path, text(exc)))
    data = sqlparse.format(data, strip_comments=True)
    for statement in sqlparse.parsestream(data, encoding='utf-8'):
        st_str = text(statement).strip().encode('utf-8')
        if st_str:
            yield st_str


def _apply_statement(statement, cursor):
    """
    Execute statement using cursor
    """
    try:
        cursor.execute(statement)
    except psycopg2.Error as exc:
        LOG.error('Error executing statement:')
        for line in statement.splitlines():
            LOG.error(line)
        LOG.error(exc)
        raise MigrateError('Unable to apply statement')


def _apply_file(file_path, cursor):
    """
    Execute all statements in file
    """
    try:
        for statement in _get_statements(file_path):
            _apply_statement(statement, cursor)
    except MalformedStatement as exc:
        LOG.error(exc)
        raise exc


def _apply_version(version, cfg, cursor):
    """
    Execute all statements in migration version
    """
    all_versions = _get_migrations_info_from_dir(cfg.base_dir)
    version_info = all_versions[version]
    LOG.info('Try apply version %r', version_info)

    _apply_file(version_info.filePath, cursor)
    query = cursor.mogrify(f'INSERT INTO public.{MIGRATIONS_TABLE_NAME} '
                           f'(version, description, installed_by) '
                           f'VALUES (%s::bigint, %s, %s)',
                           (text(version),
                            version_info.meta['description'],
                            cfg.user))
    cursor.execute(query)


def _parse_str_callbacks(callbacks, ret, base_dir):
    callbacks = callbacks.split(',')
    for callback in callbacks:
        if not callback:
            continue
        tokens = callback.split(':')
        if tokens[0] not in ret._fields:
            raise ConfigParseError('Unexpected callback '
                                   'type: %s' % text(tokens[0]))
        path = os.path.join(base_dir, tokens[1])
        if not os.path.exists(path):
            raise ConfigParseError('Path unavailable: %s' % text(path))
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                getattr(ret, tokens[0]).append(os.path.join(path, fname))
        else:
            getattr(ret, tokens[0]).append(path)

    return ret


def _parse_dict_callbacks(callbacks, ret, base_dir):
    for i in callbacks:
        if i in ret._fields:
            for j in callbacks[i]:
                path = os.path.join(base_dir, j)
                if not os.path.exists(path):
                    raise ConfigParseError('Path unavailable: %s' % text(path))
                if os.path.isdir(path):
                    for fname in sorted(os.listdir(path)):
                        getattr(ret, i).append(os.path.join(path, fname))
                else:
                    getattr(ret, i).append(path)
        else:
            raise ConfigParseError('Unexpected callback type: %s' % text(i))

    return ret


def _get_callbacks(callbacks, base_dir=''):
    """
    Parse cmdline/config callbacks
    """
    ret = Callbacks(beforeAll=[],
                    beforeEach=[],
                    afterEach=[],
                    afterAll=[])
    return _parse_dict_callbacks(callbacks, ret, base_dir) \
        if isinstance(callbacks, dict) \
        else _parse_str_callbacks(callbacks, ret, base_dir)


def _migrate_step(state, callbacks, cursor, cfg):
    """
    Apply one version with callbacks
    """
    before_all_executed = False
    should_migrate = False
    if not _is_initialized(cfg):
        LOG.info('schema not initialized')
        _init_schema(cursor)
    for version in sorted(state.keys()):
        LOG.debug('has version %r', version)
        if state[version]['installed_on'] is None:
            should_migrate = True
            if not before_all_executed and callbacks.beforeAll:
                LOG.info('Executing beforeAll callbacks:')
                for callback in callbacks.beforeAll:
                    _apply_file(callback, cursor)
                    LOG.info(callback)
                before_all_executed = True

            LOG.info('Migrating to version %d', version)
            if callbacks.beforeEach:
                LOG.info('Executing beforeEach callbacks:')
                for callback in callbacks.beforeEach:
                    LOG.info(callback)
                    _apply_file(callback, cursor)

            _apply_version(version, cfg, cursor)

            if callbacks.afterEach:
                LOG.info('Executing afterEach callbacks:')
                for callback in callbacks.afterEach:
                    LOG.info(callback)
                    _apply_file(callback, cursor)

    if should_migrate and callbacks.afterAll:
        LOG.info('Executing afterAll callbacks:')
        for callback in callbacks.afterAll:
            LOG.info(callback)
            _apply_file(callback, cursor)


def _finish(config):
    if config.dryrun:
        config.cursor.execute('rollback')
    else:
        config.cursor.execute('commit')


def info(cfg, stdout=True):
    """
    Info cmdline wrapper
    """
    state = _get_state(cfg)
    if stdout:
        out_state = OrderedDict()
        for version in sorted(state, key=int):
            out_state[version] = state[version]
        sys.stdout.write(
            json.dumps(out_state, indent=4, separators=(',', ': ')) + '\n')

    _finish(cfg)

    return state


def clean(cfg):
    """
    Drop migrations table table
    """
    if _is_initialized(cfg):
        LOG.info(f'dropping {MIGRATIONS_TABLE_NAME}')
        cfg.cursor.execute(f'DROP TABLE public.{MIGRATIONS_TABLE_NAME};')
        LOG.info(cfg.cursor.statusmessage)
        LOG.info('dropping schema_version_type')
        cfg.cursor.execute('DROP TYPE public.schema_version_type;')
        LOG.info(cfg.cursor.statusmessage)
        _finish(cfg)


def baseline(cfg):
    """
    Set baseline cmdline wrapper
    """
    if not _is_initialized(cfg):
        _init_schema(cfg.cursor)
    _set_baseline(cfg)

    _finish(cfg)


def _prepare_nontransactional_steps(state, callbacks):
    steps = []
    i = {'state': {},
         'cbs': _get_callbacks('')}
    for version in sorted(state):
        if not state[version]['transactional']:
            if i['state']:
                steps.append(i)
                i = {'state': {},
                     'cbs': _get_callbacks('')}
            elif not steps:
                LOG.error('First migration MUST be transactional')
                raise MalformedMigration('First migration MUST '
                                         'be transactional')
            steps.append({'state': {version: state[version]},
                          'cbs': _get_callbacks('')})
        else:
            i['state'][version] = state[version]
            i['cbs'] = callbacks

    if i['state']:
        steps.append(i)

    transactional = []
    for (num, step) in enumerate(steps):
        if list(step['state'].values())[0]['transactional']:
            transactional.append(num)

    if len(transactional) > 1:
        for num in transactional[1:]:
            steps[num]['cbs'] = steps[num]['cbs']._replace(beforeAll=[])
        for num in transactional[:-1]:
            steps[num]['cbs'] = steps[num]['cbs']._replace(afterAll=[])

    LOG.info('Initialization plan result:\n %s',
             json.dumps(steps, indent=4, separators=(',', ': ')))

    return steps


def migrate(cfg):
    """
    Migrate cmdline wrapper
    """
    if cfg.target is None:
        LOG.error('Unknown target (you could use "latest" to '
                  'use latest available version)')
        raise MigrateError('Unknown target')

    state = _get_state(cfg)
    not_applied = [x for x in state if state[x]['installed_on'] is None]
    non_trans = [x for x in not_applied if not state[x]['transactional']]

    if non_trans:
        if cfg.dryrun:
            LOG.error('Dry run for nontransactional migrations '
                      'is nonsence')
            raise MigrateError('Dry run for nontransactional migrations '
                               'is nonsence')
        if len(state) != len(not_applied):
            if len(not_applied) != len(non_trans):
                LOG.error('Unable to mix transactional and '
                          'nontransactional migrations')
                raise MigrateError('Unable to mix transactional and '
                                   'nontransactional migrations')
            cfg.cursor.execute('rollback;')
            nt_conn = _create_connection(cfg.conn)
            nt_conn.autocommit = True
            cursor = _init_cursor(nt_conn, cfg.session)
            _migrate_step(state, _get_callbacks(''), cursor, cfg)
        else:
            steps = _prepare_nontransactional_steps(state, cfg.callbacks)

            nt_conn = _create_connection(cfg.conn)
            nt_conn.autocommit = True

            commit_req = False
            for step in steps:
                if commit_req:
                    cfg.cursor.execute('commit')
                    commit_req = False
                if not list(step['state'].values())[0]['transactional']:
                    cur = _init_cursor(nt_conn, cfg.session)
                else:
                    cur = cfg.cursor
                    commit_req = True
                _migrate_step(step['state'], step['cbs'], cur, cfg)
    else:
        _migrate_step(state, cfg.callbacks, cfg.cursor, cfg)

    _finish(cfg)


def create_migration(cfg):
    """
    Basic migration constructor
    """
    ver, fmt = "timestamp|%Y%m%d%H%M%S".split('|')

    migration = {
        'next_version': None,
        'description': (cfg.message.replace(' ', '_')).strip()
    }

    if not migration.get('description'):
        raise RuntimeError('New migration should have a name. '
                           'Use "-m" key to set migration name.')

    if ver == 'timestamp':
        migration['next_version'] = datetime.now().strftime(fmt) \
            if cfg.message != "test0123" \
            else "0123"  # to have predictable migration file name for testing purposes

    migration_name = "V%(next_version)s__%(description)s.sql" % migration
    migration_path = os.path.join(cfg.base_dir, MIGRATION_FOLDER)

    with open(os.path.join(migration_path, migration_name), "w") as mig:
        mig.write(MIGRATION_DEFAULT_ENCODING + '\n\n')

    print("New migration: %s" % os.path.join(migration_path, migration_name))


COMMANDS = {
    'info': info,
    'clean': clean,
    'baseline': baseline,
    'migrate': migrate,
    'create': create_migration,
}

CONFIG_DEFAULTS = Config(target=None, baseline=0, cursor=None, dryrun=False,
                         callbacks='', base_dir='', user=None,
                         session=['SET lock_timeout = 0'], message='',
                         conn='dbname=postgres user=postgres',
                         conn_instance=None, table_name=MIGRATIONS_TABLE_NAME)


def get_config(base_dir, args=None):
    """
    Load configuration from yml in base dir with respect of args
    """
    global MIGRATIONS_TABLE_NAME 
    base_dir = MIGRATION_FOLDER \
        if not base_dir or base_dir == "." \
        else base_dir

    _config = MIGRATION_CONFIG \
        if not hasattr(args, "cfg") or not args.cfg \
        else args.cfg

    _config = f"migrations_{_config}" \
        if not _config.startswith("migrations") else _config

    _config += ".yml" \
        if not _config.endswith(".yml") else ""

    """
    It is possible to maintain several migration types:
      - migration_data
      - migration_schema
      - migration_testing_data
      ...
      
    This migrations can obtain their own settings depends on environment:
      - migration_data_production.yml
      - migration_data_develop.yml
      - migration_data_local_my_very_expiremental_feature.yml
      ...
    
    Thus, each `migration type` should have different tables, but 
    the same table name in different environments, 
    so let's name table like two first config parameters 
    and by default it will be just `migrations`
    """
    table_name = "_".join(_config.replace(".yml", "").split("_")[:2])
    MIGRATIONS_TABLE_NAME = table_name
    
    path = os.path.join(base_dir, _config)

    try:
        with codecs.open(path, encoding='utf-8') as i:
            base = yaml.load(i.read())
    except IOError:
        LOG.info('Unable to load %s. Using defaults', path)
        base = {}

    conf = CONFIG_DEFAULTS
    for i in [j for j in CONFIG_DEFAULTS._fields if j not in CONFIG_IGNORE]:
        if i in base:
            conf = conf._replace(**{i: base[i]})
        if args is not None:
            if i in args.__dict__ and args.__dict__[i] is not None:
                conf = conf._replace(**{i: args.__dict__[i]})

    if conf.target is not None:
        if conf.target == 'latest':
            conf = conf._replace(target=float('inf'))
        else:
            conf = conf._replace(target=int(conf.target))

    if args and args.message:
        conf = conf._replace(message=args.message)

    conf = conf._replace(conn_instance=_create_connection(conf.conn))
    conf = conf._replace(cursor=_init_cursor(conf.conn_instance, conf.session))
    conf = conf._replace(table_name=table_name) \
        if not base.get("table_name") \
        else conf._replace(table_name=base.get("table_name").strip())
    conf = conf._replace(callbacks=_get_callbacks(conf.callbacks,
                                                  conf.base_dir))

    if conf.user is None or not conf.user:
        conf = conf._replace(user=_get_database_user(conf.cursor))

    return conf


def _main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('cmd',
                        choices=COMMANDS.keys(),
                        type=str,
                        help='Operation')
    parser.add_argument('-t', '--target',
                        type=str,
                        help='Target version')
    parser.add_argument('-c', '--conn',
                        type=str,
                        help='Postgresql connection string')
    parser.add_argument('-g', '--cfg',
                        default=MIGRATION_CONFIG,
                        help='Migrations config file name')
    parser.add_argument('-d', '--base_dir',
                        type=str,
                        default='.',
                        help='Migrations base dir')
    parser.add_argument('-u', '--user',
                        type=str,
                        help='Override database user in migration info')
    parser.add_argument('-b', '--baseline',
                        type=int,
                        help='Baseline version')
    parser.add_argument('-a', '--callbacks',
                        type=str,
                        help='Comma-separated list of callbacks '
                             '(type:dir/file)')
    parser.add_argument('-s', '--session',
                        action='append',
                        help='Session setup (e.g. isolation level)')
    parser.add_argument('-n', '--dryrun',
                        action='store_true',
                        help='Say "rollback" in the end instead of "commit"')
    parser.add_argument('-v', '--verbose',
                        default=0,
                        action='count',
                        help='Be verbose')
    parser.add_argument('-m', '--message',
                        type=str,
                        help='Migration message')

    args = parser.parse_args()
    logging.basicConfig(
        level=(logging.ERROR - 10 * (min(3, args.verbose))),
        format='%(asctime)s %(levelname)-8s: %(message)s')

    config = get_config(args.base_dir, args)
    COMMANDS[args.cmd](config)


if __name__ == '__main__':
    _main()
