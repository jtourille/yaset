from setuptools import setup

setup(name='yaset',
      version='0.1',
      description='Yet Another SEquence Tagger',
      url='https://github.com/jtourille/yaset',
      author='Julien Tourille',
      author_email='julien.tourille@limsi.fr',
      license='MIT',
      packages=['yaset'],
      install_requires=[
          'scikit_learn',
          'tensorflow',
          'gensim',
          'numpy',
          'prettytable'
      ],
      scripts=['bin/yaset'],
      zip_safe=False)
