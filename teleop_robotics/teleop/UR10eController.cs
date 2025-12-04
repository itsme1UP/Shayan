using UnityEngine;
using RosSharp.RosBridgeClient;

public class UR10eController : MonoBehaviour
{
    // Robot joint position (X-axis only for 2 DOF)
    private float targetX = 0.0f;

    // ROS Subscriber
    public RosConnector rosConnector;
    public string topic = "/ur10e/cmd_vel";

    private TwistSubscriber twistSubscriber;

    void Start()
    {
        twistSubscriber = gameObject.AddComponent<TwistSubscriber>();
        twistSubscriber.Topic = topic;
        twistSubscriber.RosConnector = rosConnector;
        twistSubscriber.OnTwistReceived += OnTwist;
    }

    void OnTwist(TwistMessage msg)
    {
        // Only X linear movement
        targetX = msg.linear_x;
    }

    void Update()
    {
        // Move UR10e model along X-axis
        Vector3 pos = transform.localPosition;
        pos.x = Mathf.Lerp(pos.x, targetX, 0.2f); // smooth movement
        transform.localPosition = pos;
    }
}
