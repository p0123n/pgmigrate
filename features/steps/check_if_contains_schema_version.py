from behave import then

from pgmigrate import _is_initialized


@then("database contains schema_version")
def step_impl(context):
    class Config:
        cursor = context.conn.cursor()
        table_name = "schema_version"

    assert _is_initialized(Config), 'Non-empty db should be initialized'
