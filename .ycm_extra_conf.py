def Settings( **kwargs ):
  return {
    'flags': [
        '-x', 'c++',
        '-std=c++2a',
        '-Wall', '-Wextra', '-Werror',
        '-DAI_LOGGING_ENABLED',
        '-Iinclude/',
        '-Ideps/AI-Toolbox/include/',
        '-isystem/home/svalorzen/Projects/eigen-3.4.0-install/include',
    ],
  }
