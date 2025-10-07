from setuptools import setup, find_packages

with open('requirements-mac.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='seed',
    version='1.1.1',
    packages=find_packages(),
    url='https://github.com/Neuralgarage/seed-vc-fork-2',
    python_requires='>=3.10',
    license='MIT',
    author='Neuralgarage',
    author_email="info@neuralgarage.com",
    description='seed-vc package',
    install_requires=requirements
)