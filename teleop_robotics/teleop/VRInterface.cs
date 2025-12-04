using UnityEngine;
using Valve.VR;

public class VRInterface : MonoBehaviour
{
    public SteamVR_Input_Sources handType; // Left or Right Hand
    public SteamVR_Behaviour_Pose controllerPose;
    public SteamVR_Action_Boolean grabAction;

    void Update()
    {
        // Example: trigger grab input
        if(grabAction.GetStateDown(handType))
        {
            Debug.Log(handType + " grabbed");
        }

        // Update controller position/rotation if needed
        Vector3 pos = controllerPose.transform.position;
        Quaternion rot = controllerPose.transform.rotation;
        transform.localPosition = pos;
        transform.localRotation = rot;
    }
}
