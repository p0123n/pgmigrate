Feature: Create

    Scenario: Make a new empty migration
        Given migration dir
        And database and connection
        And successful pgmigrate run with "create -m test0123"
        When we run pgmigrate with "create -m test0123"
        Then pgmigrate command "succeeded"
        And migration dir exists
        And migration dir has migration file

    Scenario: Creation new migration fails coz migration name is not set
        Given migration dir
        And database and connection
        And migration dir has wrong access rights
        When we run pgmigrate with "create"
        Then migrate command failed with RuntimeError
