def main():
    from .cli import cli

    cli()

    from llm.plugins import pm, load_plugins

    load_plugins()

    pm.hook.register_commands(cli=cli)


if __name__ == "__main__":
    main()
