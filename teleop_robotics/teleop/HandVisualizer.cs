using UnityEngine;
using RosSharp.RosBridgeClient;

public class HandVisualizer : MonoBehaviour
{
    public RosConnector rosConnector;
    public string topic = "/leap_motion/hand";

    private HandSubscriber handSubscriber;

    void Start()
    {
        handSubscriber = gameObject.AddComponent<HandSubscriber>();
        handSubscriber.Topic = topic;
        handSubscriber.RosConnector = rosConnector;
        handSubscriber.OnHandReceived += OnHandUpdate;
    }

    void OnHandUpdate(HandMessage msg)
    {
        // Map hand position to Unity space
        Vector3 newPos = transform.localPosition;
        newPos.x = msg.x;
        newPos.y = msg.y;
        newPos.z = msg.z;
        transform.localPosition = newPos;
    }
}
