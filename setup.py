from setuptools import setup

setup(name='Payne4MIKE',
      version='1.0',
      description='Tools for fitting Magellan/MIKE spectra with The Payne.',
      author='Yuan-Sen Ting',
      author_email='ting@ias.edu',
      license='MIT',
      url='https://github.com/tingyuansen/Payne4MIKE',
      package_dir = {},
      packages=['Payne4MIKE'],
      package_data={'Payne4MIKE':['other_data/*.npz','other_data/*.fits']},
      dependency_links = [],
      install_requires=[])
