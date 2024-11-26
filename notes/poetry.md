# Poetry

[Documentation](https://python-poetry.org/docs/)

If you have existing project, you can add **poetry** to it with command

```
poetry init
```

After executing this command, `pyproject.toml` file will appear.

To add packages:

```
poetry add <package-name-1> <package-name-2> ...
```

Also, you can specify versions and place for dependencies (see more
[here](https://python-poetry.org/docs/cli#add)).

After last command `poetry.lock` file will appear. In it you can find all
dependencies and meta info for all packages. Moreover, you can build tree of
dependencies using CLI:

```
poetry show --tree
```

If you want to add some packages, you can do it by modifying `pyproject.toml`
file. After that you need to run

```
poetry lock --no-update
```

to update `poetry.lock` file. Then, execute

```
poetry install
```
