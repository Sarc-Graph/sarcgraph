## 0.3.0 - YYYY-MM-DD

### Added
- **config.py**: Introduced a new configuration file to centralize all input options and settings for the application. This file contains the `Config` class, which utilizes data classes for better structure and validation of configuration parameters. The class includes various attributes for configuration settings, along with comprehensive type annotations and validations within the `__post_init__` method.

### Changed
- **sg.py**: Migrated configuration options and input settings to `config.py`. This change simplifies `sg.py` by offloading the responsibility of handling configuration to the `Config` class in `config.py`. As a result, `sg.py` now imports the configuration settings from `config.py`, ensuring a more modular and cleaner code structure.

### Deprecated
- Direct use of configuration options within `sg.py`. Users are encouraged to update their workflow to utilize the `Config` class from `config.py`.

### Fixed
- Ensured that all configuration parameters are validated for correct data types and logical consistency upon initialization. This validation is handled by the `__post_init__` method within the `Config` class in `config.py`.

### Removed
- Redundant code in `sg.py` related to handling configuration settings, as these responsibilities have been transferred to `config.py`.