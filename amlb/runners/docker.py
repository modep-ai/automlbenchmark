"""
**docker** module is build on top of **ContainerBenchmark** module to provide logic to create and run docker images
that are preconfigured with a given automl framework, and that can be used to run a benchmark anywhere.
The docker image embeds a version of the automlbenchmark app so that tasks are later run in local mode inside docker,
providing the same parameters and features allowing to import config and export results through mounted folders.
"""
import logging
import os
import re

from ..resources import config as rconfig, get as rget
from ..utils import dir_of, run_cmd, str_digest, str_sanitize, touch
from .container import ContainerBenchmark


log = logging.getLogger(__name__)


class DockerBenchmark(ContainerBenchmark):
    """DockerBenchmark
    an extension of ContainerBenchmark to run benchmarks inside docker.
    """

    def __init__(self, framework_name, benchmark_name, constraint_name):
        """

        :param framework_name:
        :param benchmark_name:
        :param constraint_name:
        """
        super().__init__(framework_name, benchmark_name, constraint_name)
        self._custom_image_name = rconfig().docker.image
        self.minimize_instances = rconfig().docker.minimize_instances
        self.container_name = 'docker'
        self.force_branch = rconfig().docker.force_branch
        self.custom_commands = self.framework_module.docker_commands(
            self.framework_def.setup_args,
            setup_cmd=self.framework_def._setup_cmd
        ) if hasattr(self.framework_module, 'docker_commands') else ""

    @property
    def _script(self):
        return os.path.join(self._framework_dir, 'Dockerfile')

    def _start_container(self, script_params=""):
        """Implementes the container run method"""
        in_dir = rconfig().input_dir
        out_dir = rconfig().output_dir
        custom_dir = rconfig().user_dir
        for d in [in_dir, out_dir, custom_dir]:
            touch(d, as_dir=True)
        script_extra_params = "--session="  # in combination with `self.output_dirs.session` usage below to prevent creation of 2 sessions locally
        inst_name = f"{self.sid}.{str_sanitize(str_digest(script_params))}"
        cmd = (
            "docker run --name {name} {options} "
            "-v {input}:/input -v {output}:/output -v {custom}:/custom "
            "--rm {image} {params} -i /input -o /output -u /custom -s skip -Xrun_mode=docker {extra_params}"
        ).format(
            name=inst_name,
            options=rconfig().docker.run_extra_options,
            input=in_dir,
            output=self.output_dirs.session,
            custom=custom_dir,
            image=self.image,
            params=script_params,
            extra_params=script_extra_params,
        )
        log.info("Starting docker: %s.", cmd)
        log.info("Datasets are loaded by default from folder %s.", in_dir)
        log.info("Generated files will be available in folder %s.", out_dir)
        try:
            run_cmd(cmd, _capture_error_=False)  # console logs are written on stderr by default: not capturing allows live display
        except:  # also want to handle KeyboardInterrupt
            try:
                run_cmd(f"docker kill {inst_name}")
            except Exception:
                pass
            finally:
                raise

    def _image_exists(self, image):
        """Implements a method to see if the container image is available"""
        output, _ = run_cmd(f"docker images -q {image}")
        log.debug("docker image id: %s", output)
        if re.match(r'^[0-9a-f]+$', output.strip()):
            return True
        try:
            run_cmd(f"docker pull {image}", _live_output_=True)
            return True
        except Exception:
            pass
        return False

    def _run_container_build_command(self, image, cache):
        log.info(f"Building docker image {image}.")
        run_cmd("docker build {options} -t {container} -f {script} .".format(
            options="" if cache else "--no-cache",
            container=image,
            script=self._script),
            _live_output_=rconfig().setup.live_output,
            _activity_timeout_=rconfig().setup.activity_timeout
        )
        log.info(f"Successfully built docker image {image}.")

    def _upload_image(self, image):
        log.info(f"Publishing docker image {image}.")
        run_cmd(f"docker login && docker push {image}")
        log.info(f"Successfully published docker image {image}.")

    def _generate_script(self, custom_commands):
        docker_content = """FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get -y install apt-utils dialog locales
RUN apt-get -y install curl wget unzip git
RUN apt-get -y install software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python{pyv} python{pyv}-venv python{pyv}-dev python3-pip
#RUN update-alternatives --install /usr/bin/python3 python3 $(which python{pyv}) 1

# aliases for the python system
ENV SPIP python{pyv} -m pip
ENV SPY python{pyv}

# Enforce UTF-8 encoding
ENV PYTHONUTF8 1
ENV PYTHONIOENCODING utf-8
# RUN locale-gen en-US.UTF-8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /bench

# We create a virtual environment so that AutoML systems may use their preferred versions of
# packages that we need to data pre- and postprocessing without breaking it.
RUN $SPIP install -U pip wheel
RUN $SPY -m venv venv
ENV PIP /bench/venv/bin/python3 -m pip
ENV PY /bench/venv/bin/python3 -W ignore
#RUN $PIP install -U pip=={pipv} wheel
RUN $PIP install -U pip wheel

VOLUME /input
VOLUME /output
VOLUME /custom

# Add the AutoML system except files listed in .dockerignore (could also use git clone directly?)
ADD . /bench/

RUN (grep -v '^\\s*#' | xargs -L 1 $PIP install --no-cache-dir) < requirements.txt

RUN $PY {script} {framework} -s only
{custom_commands}

# https://docs.docker.com/engine/reference/builder/#entrypoint
ENTRYPOINT ["/bin/bash", "-c", "$PY {script} $0 $*"]
CMD ["{framework}", "test"]

""".format(
            custom_commands=custom_commands.format(
                setup=dir_of(os.path.join(self._framework_dir, "setup/"),
                             rel_to_project_root=True),
                pip="$PIP",
                py="$PY"
            ),
            framework=self._forward_params['framework_name'],
            pyv=rconfig().versions.python,
            pipv=rconfig().versions.pip,
            script=rconfig().script,
            user=rconfig().user_dir,
        )
        with open(self._script, 'w') as file:
            file.write(docker_content)

class DockerBenchmarkAPI(DockerBenchmark):
    """DockerBenchmarkAPI
    an extension of DockerBenchmark to run benchmarks inside docker by exposing a FLASK API.
    """
    def _generate_script(self, custom_commands):
        docker_content = """FROM ubuntu:18.04

RUN mkdir -p /etc/supervisor/conf.d/
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY ./my_redis.conf /etc/redis.conf

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get -y install apt-utils dialog locales
RUN apt-get -y install curl wget unzip git nano
RUN apt-get -y install software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python{pyv} python{pyv}-venv python{pyv}-dev python3-pip
#RUN update-alternatives --install /usr/bin/python3 python3 $(which python{pyv}) 1

# aliases for the python system
ENV SPIP python{pyv} -m pip
ENV SPY python{pyv}

# Enforce UTF-8 encoding
ENV PYTHONUTF8 1
ENV PYTHONIOENCODING utf-8
# RUN locale-gen en-US.UTF-8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /bench

# We create a virtual environment so that AutoML systems may use their preferred versions of
# packages that we need to data pre- and postprocessing without breaking it.
RUN $SPIP install -U pip wheel
RUN $SPY -m venv venv
ENV PIP /bench/venv/bin/python3 -m pip
ENV PY /bench/venv/bin/python3 -W ignore
#RUN $PIP install -U pip=={pipv} wheel
RUN $PIP install -U pip wheel

VOLUME /input
VOLUME /output
VOLUME /custom

# ---------------------------------------------------------------
# above here is the same as DockerBenchmark
# ---------------------------------------------------------------

# Need libpq-dev to pip install the psycopg2 package.
# Need redis-server for celery.
RUN apt-get -y install libpq-dev redis-server supervisor

RUN $PIP install -U google-cloud-storage

# Add the AutoML system except files listed in .dockerignore (could also use git clone directly?)
ADD automlbenchmark /bench/automlbenchmark
ADD worker_app /bench/worker_app
ADD app_utils /bench/app_utils

# remove artifacts from local installations of frameworks created by `-m local`
RUN rm -rf automlbenchmark/frameworks/*/venv
RUN rm -rf automlbenchmark/frameworks/*/.installed
RUN rm -rf automlbenchmark/frameworks/*/.setup_env

# WARNING: do not distribute the container since it now has this key in it
RUN mkdir -p /var/secrets
COPY ./gcp_key.json /var/secrets/gcp_key.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/gcp_key.json

# install AMLB in the same order as requirements.txt
RUN (grep -v '^\\s*#' | xargs -L 1 $PIP install --no-cache-dir) < /bench/automlbenchmark/requirements.txt

# install common app_utils
RUN $PIP install --no-cache-dir -r /bench/app_utils/requirements.txt
RUN $PIP install --no-cache-dir /bench/app_utils/

# install worker_app
RUN $PIP install --no-cache-dir -r /bench/worker_app/requirements.txt
RUN $PIP install --no-cache-dir /bench/worker_app/

# run setup.sh script for this framework
# $PY /bench/automlbenchmark/{script} {framework} -s only

RUN $PY /bench/automlbenchmark/runbenchmark.py autogluon -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py autosklearn -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py autoweka -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py flaml -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py gama -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py h2oautoml -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py hyperoptsklearn -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py mljarsupervised -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py mlnet -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py tpot -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py constantpredictor -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py randomforest -s only
RUN $PY /bench/automlbenchmark/runbenchmark.py tunedrandomforest -s only

# Custom commands
{custom_commands}

EXPOSE 8080

ENV GCP_BUCKET={GCP_BUCKET}
ENV DB_HOST={DB_HOST}
ENV DB_PORT={DB_PORT}
ENV DB_USER={DB_USER}
ENV DB_PASS={DB_PASS}
ENV DB_DB={DB_DB}

# RUN redis-server &
# RUN celery -A worker_app.celery worker -l debug &

# start API server using the opened port above (8080).
# # CMD ["/bench/venv/bin/python3", "/bench/worker_app/worker_app/app.py", "-d", "{DEBUG_SERVER}", "--port", "8080", "--host", "0.0.0.0"]

# uses 0.0.0.0/8080 for the web server in supervisord.conf
CMD ["supervisord"]
""".format(
    custom_commands=custom_commands.format(
        setup=dir_of(os.path.join(self._framework_dir, "setup/"),
                     rel_to_project_root=True),
        pip="$PIP",
        py="$PY"
    ),
    framework=self._forward_params['framework_name'],
    pyv=rconfig().versions.python,
    pipv=rconfig().versions.pip,
    script=rconfig().script,
    user=rconfig().user_dir,
    GCP_BUCKET=os.environ.get('GCP_BUCKET', ''),
    DEBUG_SERVER=os.environ.get('DEBUG_SERVER', '1'),
    DB_HOST=os.environ.get('DB_HOST', 'localhost'),
    DB_PORT=os.environ.get('DB_PORT', '5432'),
    DB_USER=os.environ.get('DB_USER', 'postgres'),
    DB_PASS=os.environ.get('DB_PASS', ''),
    DB_DB=os.environ.get('DB_DB', 'mlapi'),
)
        with open(self._script, 'w') as file:
            file.write(docker_content)
