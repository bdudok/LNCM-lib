from LG_API_token import baserow_token
config = {
  "http": "http://10.18.4.112:5001",
  "api_token": baserow_token,
  "session_url": "http://10.18.4.112:5001/api/database/rows/table/490/?user_field_names=true",
  "mice_url": "http://10.18.4.112:5001/api/database/rows/table/492/?user_field_names=true",
  "FieldID": {
      'Task': 4639,
      'Mouse.ID': 4616,
      'InclTag': 4519,
      'Project': 4637
  },
  "lg_api_root": "https://my.labguru.com/",
  "lg_mice_url": "https://my.labguru.com/api/v1/biocollections/Mice",
  "lg_protocol_url": "https://my.labguru.com/api/v1/protocols",
}
#api_token is read-only token with Barna account