from setuptools import setup, find_packages
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]


setup(
    name='pyhurricane',
    version='0.1.0',
    description='Analysis programs for tropical cyclone by gpv data',
    author='Hiroaki Yoshioka',
    author_email='y0sh10ka.h.1030@gmail.com',
    url='https://github.com/H1r0ak1Y0sh10ka/pyhurricane',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
)