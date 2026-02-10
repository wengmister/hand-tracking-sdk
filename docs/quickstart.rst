Quickstart
==========

This page shows minimal usage examples for parsing and streaming.

Parsing One Line
----------------

.. code-block:: python

   from hand_tracking_sdk import parse_line

   packet = parse_line("Right wrist:, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0")
   print(packet.side, packet.kind, packet.data)

Streaming Frames
----------------

.. code-block:: python

   from hand_tracking_sdk import HTSClient, HTSClientConfig, StreamOutput

   client = HTSClient(
       HTSClientConfig(
           output=StreamOutput.FRAMES,
           host="0.0.0.0",
           port=9000,
       )
   )

   for event in client.iter_events():
       print(event)

Coordinate Conversion
---------------------

.. code-block:: python

   from hand_tracking_sdk import convert_hand_frame_unity_left_to_right

   converted = convert_hand_frame_unity_left_to_right(event)

Notes
-----

- HTS data uses Unity left-handed coordinates.
- For many robotics stacks, convert data to right-handed coordinates.
- In strict mode, malformed lines raise ``ParseError``.
