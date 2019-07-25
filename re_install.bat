call activate env_36

python setup.py bdist_wheel --universal

pip uninstall ps -y

pip install dist\ps-0.0.1-py2.py3-none-any.whl

::twine upload --repository-url http://192.168.1.195:8071/repository/jasperpypi-hosted/ dist\jt.app-0.1.6-py2.py3-none-any.whl -u jasper -p jasper123