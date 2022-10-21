from setuptools import find_packages, setup

packages = find_packages(exclude=("setup", "tests"))

version = "0.0.1"

setup(
    name="mup_tf",
    version=version,
    description="Maximal Update Parametrization in Tensorflow",
    author="Zach Nussbaum",
    author_email="zanussbaum@gmail.com",
    url="https://github.com/zanussbaum/mup-tf",
    packages=packages,
    include_package_data=True,  # for manifest files.
    python_requires=">=3.7, <4",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=[
        "tensorflow>=2.6.2",
        "tqdm",
        "pyyaml",
        "matplotlib",
        "seaborn",
    ],
    license="PROPRIETARY",
    test_suite="tests",
    tests_require=["pytest-runner"],
)
