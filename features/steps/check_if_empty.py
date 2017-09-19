from behave import then

from pgmigrate import _is_initialized


@then("database has no schema_version table")
def step_impl(context):
    class Config:
        cursor = context.conn.cursor()
        table_name = "schema_version"

    assert not _is_initialized(Config), 'Database should be uninitialized'
