import pip
from setuptools import setup, find_packages


def is_gpu():
    packages = [str(i) for i in pip.get_installed_distributions()]
    for item in packages:
        if "tensorflow-gpu<=1.3.0" in item:
            return True
    return False


kw = [
    'scikit_learn',
    'gensim',
    'numpy',
    'prettytable',
    'tensorflow<=1.3.0'
]

try:
    import tensorflow
except ImportError:
    kw.pop()
    kw.append('tensorflow<=1.3.0')
else:
    if is_gpu():
        kw.pop()
        kw.append('tensorflow-gpu<=1.3.0')
    else:
        kw.pop()
        kw.append('tensorflow<=1.3.0')

setup(name='YASET',
      version='0.3',
      description='Yet Another SEquence Tagger',
      url='https://github.com/jtourille/yaset',
      author='Julien Tourille',
      author_email='julien.tourille@limsi.fr',
      license='GPL-3.0',
      packages=find_packages(),
      package_data={'yaset': [
          'desc/BILSTMCHARCRF_PARAMS_DESC.json',
          'desc/DATA_PARAMS_DESC.json',
          'desc/GENERAL_PARAMS_DESC.json',
          'desc/TRAINING_PARAMS_DESC.json'
      ]},
      scripts=['bin/yaset'],
      zip_safe=False,
      install_requires=kw)
