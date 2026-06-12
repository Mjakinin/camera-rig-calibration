# Station Suitability

| Station | Role | Static status | Moving status | Usable as | Reason |
|---|---|---|---|---|---|
| F1 | front | pose_valid | failed_not_enough_markers | not_usable | moving_calib_camera is not pose_valid / not enough station markers |
| F2 | front | pose_valid | failed_not_enough_markers | not_usable | moving_calib_camera is not pose_valid / not enough station markers |
| F3 | front | pose_valid | pose_valid | front_anchor | static camera and moving camera both have pose_valid for this station |
| F4 | front | pose_valid | pose_valid | front_anchor | static camera and moving camera both have pose_valid for this station |
| G | floor/general |  | pose_valid | not_used_floor_or_general_case | floor/general limit case; not used for front-rear chain |
| R1 | rear | pose_valid | pose_valid | rear_anchor | static camera and moving camera both have pose_valid for this station |
| R2 | rear | pose_valid | failed_not_enough_markers | not_usable | moving_calib_camera is not pose_valid / not enough station markers |
| R3 | rear | pose_valid | pose_valid | rear_anchor | static camera and moving camera both have pose_valid for this station |
