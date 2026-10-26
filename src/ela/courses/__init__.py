"""Course files and knowledge-base state, shared by the backoffice and the CLI.

* ``ela.courses.catalog`` — the ``data/latex/{level}/{course}/`` tree: listing,
  creation, uploads (files or zip), deletion, moving a course to another level.
* ``ela.courses.state`` — what the vector store holds, per level and course,
  compared with the files on disk.
"""
