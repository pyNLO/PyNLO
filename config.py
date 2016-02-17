# -*- coding: utf-8 -*-

import sys, os
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
templates_path = ['/home/docs/checkouts/readthedocs.org/readthedocs/templates/sphinx', 'templates', '_templates', '.templates']
source_suffix = '.rst'
master_doc = 'index'
project = u'PyNLO'
copyright = u''
version = 'latest'
release = 'latest'
exclude_patterns = ['_build']
pygments_style = 'sphinx'
htmlhelp_basename = 'pynlo'
file_insertion_enabled = False
latex_documents = [
  ('index', 'pynlo.tex', u'PyNLO Documentation',
   u'', 'manual'),
]
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    if sys.version[0] == '3':  # Python 3
        from unittest.mock import MagicMock
    elif sys.version[0] == '2':  # Python 2
        from mock import Mock as MagicMock
    else:
        raise ImportError("Don't know how to import MagicMock.")

    class Mock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
                return Mock()

    MOCK_MODULES = ['pyfftw', 'scipy', 'numpy', 'matplotlib', 'matplotlib.pyplot']
	print "Mocking ", MOCK_MODULES
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


###########################################################################
#          auto-created readthedocs.org specific configuration            #
###########################################################################


#
# The following code was added during an automated build on readthedocs.org
# It is auto created and injected for every build. The result is based on the
# conf.py.tmpl file found in the readthedocs.org codebase:
# https://github.com/rtfd/readthedocs.org/blob/master/readthedocs/doc_builder/templates/doc_builder/conf.py.tmpl
#


import sys
import os.path
from six import string_types

from sphinx import version_info

from recommonmark.parser import CommonMarkParser

# Only Sphinx 1.3+
if version_info[0] == 1 and version_info[1] > 2:

    # Markdown Support
    if 'source_suffix' in globals():
        if isinstance(source_suffix, string_types) and source_suffix != '.md':
            source_suffix = [source_suffix, '.md']
        elif '.md' not in source_suffix:
            source_suffix.append('.md')
    else:
        source_suffix = ['.rst', '.md']

    if 'source_parsers' in globals():
        if '.md' not in source_parsers:
            source_parsers['.md'] = CommonMarkParser
    else:
        source_parsers = {
            '.md': CommonMarkParser,
        }

if globals().get('source_suffix', False):
    if isinstance(source_suffix, string_types):
        SUFFIX = source_suffix
    else:
        SUFFIX = source_suffix[0]
else:
    SUFFIX = '.rst'

#Add RTD Template Path.
if 'templates_path' in globals():
    templates_path.insert(0, '/home/docs/checkouts/readthedocs.org/readthedocs/templates/sphinx')
else:
    templates_path = ['/home/docs/checkouts/readthedocs.org/readthedocs/templates/sphinx', 'templates', '_templates',
                      '.templates']

# Add RTD Static Path. Add to the end because it overwrites previous files.
if not 'html_static_path' in globals():
    html_static_path = []
if os.path.exists('_static'):
    html_static_path.append('_static')
html_static_path.append('/home/docs/checkouts/readthedocs.org/readthedocs/templates/sphinx/_static')

# Add RTD Theme only if they aren't overriding it already
using_rtd_theme = False
if 'html_theme' in globals():
    if html_theme in ['default']:
        # Allow people to bail with a hack of having an html_style
        if not 'html_style' in globals():
            import sphinx_rtd_theme
            html_theme = 'sphinx_rtd_theme'
            html_style = None
            html_theme_options = {}
            if 'html_theme_path' in globals():
                html_theme_path.append(sphinx_rtd_theme.get_html_theme_path())
            else:
                html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

            using_rtd_theme = True
else:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_style = None
    html_theme_options = {}
    if 'html_theme_path' in globals():
        html_theme_path.append(sphinx_rtd_theme.get_html_theme_path())
    else:
        html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    using_rtd_theme = True

# Force theme on setting
if globals().get('RTD_NEW_THEME', False):
    html_theme = 'sphinx_rtd_theme'
    html_style = None
    html_theme_options = {}
    using_rtd_theme = True

if globals().get('RTD_OLD_THEME', False):
    html_style = 'rtd.css'
    html_theme = 'default'

if globals().get('websupport2_base_url', False):
    websupport2_base_url = 'https://readthedocs.org//websupport'
    if 'http' not in settings.MEDIA_URL:
        websupport2_static_url = 'https://media.readthedocs.org/static/'
    else:
        websupport2_static_url = 'https://media.readthedocs.org//static'


#Add project information to the template context.
context = {
    'using_theme': using_rtd_theme,
    'html_theme': html_theme,
    'current_version': "latest",
    'MEDIA_URL': "https://media.readthedocs.org/",
    'PRODUCTION_DOMAIN': "readthedocs.org",
    'versions': [
    ("latest", "/en/latest/"),
    ],
    'downloads': [ 
    ],
    'slug': 'pynlo',
    'name': u'PyNLO',
    'rtd_language': u'en',
    'canonical_url': 'http://pynlo.readthedocs.org/en/latest/',
    'analytics_code': '',
    'single_version': False,
    'conf_py_path': '/./',
    'api_host': 'https://readthedocs.org/',
    'github_user': 'ycasg',
    'github_repo': 'PyNLO',
    'github_version': 'master',
    'display_github': True,
    'bitbucket_user': 'None',
    'bitbucket_repo': 'None',
    'bitbucket_version': 'master',
    'display_bitbucket': False,
    'READTHEDOCS': True,
    'using_theme': (html_theme == "default"),
    'new_theme': (html_theme == "sphinx_rtd_theme"),
    'source_suffix': SUFFIX,
    'user_analytics_code': '',
    'global_analytics_code': 'UA-17997319-1',
    
    'commit': '180d51c6',
    
}
if 'html_context' in globals():
    html_context.update(context)
else:
    html_context = context

# Add custom RTD extension
if 'extensions' in globals():
    extensions.append("readthedocs_ext.readthedocs")
else:
    extensions = ["readthedocs_ext.readthedocs"]
