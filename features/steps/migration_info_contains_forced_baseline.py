from behave import then

from pgmigrate import _get_info


@then("migration info contains forced baseline={baseline}")
def step_impl(context, baseline):
    class Config:
        cursor = context.conn.cursor()
        baseline = 0
        base_dir = context.migr_dir
        target = 1

    info = _get_info(Config)
    assert list(info.values())[0]['version'] == int(baseline)
    assert list(info.values())[0]['description'] == 'Forced baseline'
