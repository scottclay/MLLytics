from setuptools import setup

VERSION = '0.1.1'

setup(
	name='MLLytics',
    version=VERSION,
	py_modules=['MLLytics'],
    url='https://github.com/scottclay/MLLytics',
	license='MIT',
    author='Scott Clay',
    author_email='scottclay8@gmail.com',
    description='A library of tools for easier evaluation of ML models.',
	long_description=open('README.md').read(),
	long_description_content_type='text/markdown',
#	install_requires=['numpy>=',matplotlib, seaborne, pandas]
    zip_safe=False,
	classifiers = [
	'Programming Language :: Python :: 3'
#	'Topic :: Software Development :: Libaries :: Python Modules'
	]
	
	)