FINAL RESULTS FOR REPORT / MEETING
==================================

AP01:
- Ref14-origin marker relay baseline.
- Uses measured Ref14->anchor-camera ArUco/PnP and cam3-rooted AP01 relay transforms.

AP02:
- Full marker/camera graph BA.
- GT-aligned full-map evaluation.
- Reports cameras and markers 0-14. Marker14 is held out from alignment.

AP03:
- Targetless COLMAP reconstruction.
- AP03a: Single Ref14 Sim(3) scale/frame registration.
- AP03b: Multi-ArUco Sim(3) scale/frame registration.
- AP03b is the main AP03 result because it gives better static camera accuracy.

Main comparison:
Use static camera errors for cam_edge_0, cam_edge_1, cam_edge_3, cam_edge_5.
