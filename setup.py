from setuptools import setup, find_packages

setup(
    name='musedvg',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # your dependencies go here, e.g.,
        'numpy',
        'scipy'
        # etc...
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A short description',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/musedvg',
    classifiers=[
        # classifiers go here, e.g.,
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # etc...
    ]
)

