# General Structure

Metadata is serialized into a "beam." Each "beam" consists of possible training 
examples that are called "bricks". The image/video data itself is encoded into hdf5 for fast indexing.

The following is the structure for a beam:

```json
{
  "name": "NAME",
  "id": "ID",

  # File path information
  "filepath": "FILEPATH",
  "sourcePath": "SOURCEPATH",
  "sourceJsonPath": "SOURCEPATH",
  "hdf5Path": "PATH TO HDF5",
  "datasetId": "ID"

  # Video information
  "resolution": [0, 0],
  "bricks": [{}]
}
```

The following is the structure for a brick:

```json
{
  "frame": 0,
  "truth": {},
  "metadata": {},
  "rawMetadata": {}
}
```