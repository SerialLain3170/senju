# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: hello.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='hello.proto',
  package='',
  syntax='proto3',
  serialized_options=b'Z\030example.com/package/name',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0bhello.proto\"&\n\nImgMessage\x12\x0b\n\x03src\x18\x01 \x01(\t\x12\x0b\n\x03ref\x18\x02 \x01(\t\"\x1a\n\x0bImgResponse\x12\x0b\n\x03img\x18\x01 \x01(\t27\n\x05Hello\x12.\n\x0fImageManupilate\x12\x0b.ImgMessage\x1a\x0c.ImgResponse\"\x00\x42\x1aZ\x18\x65xample.com/package/nameb\x06proto3'
)




_IMGMESSAGE = _descriptor.Descriptor(
  name='ImgMessage',
  full_name='ImgMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='src', full_name='ImgMessage.src', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ref', full_name='ImgMessage.ref', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=15,
  serialized_end=53,
)


_IMGRESPONSE = _descriptor.Descriptor(
  name='ImgResponse',
  full_name='ImgResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='img', full_name='ImgResponse.img', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=55,
  serialized_end=81,
)

DESCRIPTOR.message_types_by_name['ImgMessage'] = _IMGMESSAGE
DESCRIPTOR.message_types_by_name['ImgResponse'] = _IMGRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ImgMessage = _reflection.GeneratedProtocolMessageType('ImgMessage', (_message.Message,), {
  'DESCRIPTOR' : _IMGMESSAGE,
  '__module__' : 'hello_pb2'
  # @@protoc_insertion_point(class_scope:ImgMessage)
  })
_sym_db.RegisterMessage(ImgMessage)

ImgResponse = _reflection.GeneratedProtocolMessageType('ImgResponse', (_message.Message,), {
  'DESCRIPTOR' : _IMGRESPONSE,
  '__module__' : 'hello_pb2'
  # @@protoc_insertion_point(class_scope:ImgResponse)
  })
_sym_db.RegisterMessage(ImgResponse)


DESCRIPTOR._options = None

_HELLO = _descriptor.ServiceDescriptor(
  name='Hello',
  full_name='Hello',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=83,
  serialized_end=138,
  methods=[
  _descriptor.MethodDescriptor(
    name='ImageManupilate',
    full_name='Hello.ImageManupilate',
    index=0,
    containing_service=None,
    input_type=_IMGMESSAGE,
    output_type=_IMGRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_HELLO)

DESCRIPTOR.services_by_name['Hello'] = _HELLO

# @@protoc_insertion_point(module_scope)