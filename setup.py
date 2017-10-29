import pip
from setuptools import setup, find_packages


def is_gpu():
    packages = [str(i) for i in pip.get_installed_distributions()]
    for item in packages:
        if "tensorflow-gpu" in item:
            return True
    return False


kw = [
    'scikit_learn',
    'gensim',
    'numpy',
    'prettytable'
]

try:
    import tensorflow
except ImportError:
    kw.append('tensorflow')
else:
    if is_gpu():
        kw.append('tensorflow-gpu')
    else:
        kw.append('tensorflow')

setup(name='yaset',
      version='0.1',
      description='Yet Another SEquence Tagger',
      url='https://github.com/jtourille/yaset',
      author='Julien Tourille',
      author_email='julien.tourille@limsi.fr',
      license='MIT',
      packages=find_packages(),
      package_data={'yaset': ['desc/*.json']},
      scripts=['bin/yaset'],
      zip_safe=False,
      install_requires=kw)
