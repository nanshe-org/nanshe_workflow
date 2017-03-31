[![Build Status]( https://app.wercker.com/status/1bc8209f1390ba561c535a4628c5516e/s/master "wercker status" )]( https://app.wercker.com/project/byKey/1bc8209f1390ba561c535a4628c5516e )
[![License]( https://img.shields.io/github/license/nanshe-org/nanshe_workflow.svg "license" )]( https://raw.githubusercontent.com/nanshe-org/nanshe_workflow/master/LICENSE.txt )
[![Release]( https://img.shields.io/github/release/nanshe-org/nanshe_workflow.svg "release" )]( https://github.com/nanshe-org/nanshe_workflow/releases/latest )

# Nanshe Workflow

[![Join the chat at https://gitter.im/nanshe-org/nanshe_workflow](https://badges.gitter.im/nanshe-org/nanshe_workflow.svg)](https://gitter.im/nanshe-org/nanshe_workflow?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## About

Provides a [Jupyter]( http://jupyter.org/ ) workflow for running [nanshe]( https://github.com/nanshe-org/nanshe ).

## Requirements

There are several requirements that must be satisfied to run the workflow. Also there are some optional requirements for some extensions or better performance. A list of the direct requirements can be found in the included [recipe]( ./nanshe_workflow.recipe/meta.yaml ). Basically all of these dependencies are available from [conda-forge]( https://conda-forge.github.io ) with the exception of `nanshe`, which is available from the `nanshe` channel.

## Installation

The preferred method of installation is to use [Docker]( http://docker.com/ ) as explained below. This supports all major OSes and is the easiest way to get started. In some cases, this might not be possible or desirable, in which case [conda]( http://conda.pydata.org/docs/ ) can be used to install natively.

### Docker

The easiest way to get started with the workflow is to use [docker]( http://docker.com/ ). It is available for all major OSes and is easy to install. Once installed, we provide [directions]( https://github.com/nanshe-org/docker_nanshe_workflow#standard-use ) for getting the container up and running that have very simple requirements.

### Conda

If one would rather run the workflow natively instead of using Docker, we provide an alternative installation method for Mac and Linux. Windows is currently not supported. Simply install [Miniconda]( http://conda.pydata.org/miniconda ) or [Anaconda]( https://store.continuum.io/cshop/anaconda ) based on your preference. Nearly all of the dependencies are available from the [conda-forge]( http://conda-forge.github.io ) channel. To add it simply run `conda config --add channels conda-forge`. The rest of the dependencies are in the `nanshe` channel. It can be added in the same manner `conda config --add channels nanshe`. Once done simply use [`conda-build`]( http://conda.pydata.org/docs/building/recipe.html ) to build the included metapackage by running `conda build nanshe_workflow.recipe`. Once this is complete install into your current environment by running `conda install --use-local nanshe_workflow` and you should be ready to go.

## Usage

These instructions are primarily for usage with `conda`. For typical usage with the Docker container, please see these [instructions]( https://github.com/nanshe-org/docker_nanshe_workflow#standard-use ).

### Starting

To start up the workflow, simply open a terminal and run `jupyter notebook nanshe_ipython.ipynb`. This will open a new tab/window in your browser with the workflow loaded.

### Running

Each cell with user input has an explanation beforehand with the relevant parameters noted and how they can be set. The second cell should be used to provide input. Later cells are used to specify various relevant parameters. To run a cell, just press the key combination `Shift + Enter`. This will take you to the next cell as well. Some cells may also show interactive GUIs so as to explore the result from each operation.

## Development

If you intend to work on the workflow, then you must install `jq` and have it on your path. A package for it exists in conda-forge. After cloning the repo, make sure to include the `.gitconfig` file in your `.git/config`. If just cloning the workflow, this should be as simple as running `echo -e "[include]\n\tpath = ../.gitconfig" >> .git/config`. This only needs to be done once.
