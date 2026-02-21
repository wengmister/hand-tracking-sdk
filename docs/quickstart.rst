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

Accessing Specific Joint Telemetry
----------------------------------

.. code-block:: python

   from hand_tracking_sdk import HTSClient, HTSClientConfig, JointName, StreamOutput

   client = HTSClient(HTSClientConfig(output=StreamOutput.FRAMES))

   for frame in client.iter_events():
       x, y, z = frame.get_joint(JointName.INDEX_TIP)
       print(f"index tip xyz=({x:.5f}, {y:.5f}, {z:.5f})")
       index_joints = frame.get_finger("index")
       print(index_joints[JointName.INDEX_PROXIMAL])

Streaming Frames Over TCP
-------------------------

By default, :class:`hand_tracking_sdk.HTSClient` listens in UDP mode.
If your HTS setup uses TCP, configure ``transport_mode`` explicitly as shown below.

TCP server mode (recommended for HTS wired ADB reverse and wireless TCP when
HTS is configured to connect to your host):

.. code-block:: python

   from hand_tracking_sdk import HTSClient, HTSClientConfig, StreamOutput, TransportMode

   client = HTSClient(
       HTSClientConfig(
           transport_mode=TransportMode.TCP_SERVER,
           host="0.0.0.0",
           port=8000,
           timeout_s=1.0,
           output=StreamOutput.FRAMES,
       )
   )

   for frame in client.iter_events():
       print(frame)

For VR telemetry ingestion, use ``udp`` (default) or ``tcp_server``.

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
