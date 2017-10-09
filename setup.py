from setuptools import setup

setup(name='yaset',
      version='0.1',
      description='Yet Another SEquence Tagger',
      url='https://github.com/jtourille/yaset',
      author='Julien Tourille',
      author_email='julien.tourille@limsi.fr',
      license='MIT',
      packages=['yaset'],
      package_data={'yaset': ['desc/*.json']},
      install_requires=[
          'scikit_learn',
          'tensorflow-gpu==1.2.0',
          'gensim',
          'numpy',
          'prettytable'
      ],
      scripts=['bin/yaset'],
      zip_safe=False)
