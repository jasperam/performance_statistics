"""
@author Neo
@time 2018/10/09
"""

from setuptools import setup, find_packages

setup(
    name='ps',
    version='0.1.21',
    description="performace statistics",
    author='Neo',
    author_email='neo.lin@jaspercapital.com',
    python_requires='>=3.6',
    packages=find_packages('src'),
    package_dir={"": "src"},
    package_data={
         'ps': [
             'cfg/*.ini',
             'cfg/*.yaml',
             'data/*.csv',
         ]
    },
    install_requires=[        
        'jt',
        'pandas', 
        'pyyaml',
        'pymssql',
    ],
    zip_safe=False,
)
