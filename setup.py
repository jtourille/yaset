from setuptools import setup

setup(name='yaner',
      version='0.1',
      description='Yet Another Named Entity Recognizer',
      url='https://github.com/jtourille/yaner',
      author='Julien Tourille',
      author_email='julien.tourille@limsi.fr',
      license='MIT',
      packages=['yaner'],
      install_requires=[
          'scikit_learn',
      ],
      scripts=['bin/yaner'],
      zip_safe=False)
