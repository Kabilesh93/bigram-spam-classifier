import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bigram_spam_classifier",
    version="0.0.5",
    author="Kabilesh",
    author_email="",
    description="A bigram approach for classifying Spam and Ham messages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kabilesh93/bigram-spam-classifier",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["nltk", "pandas"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
