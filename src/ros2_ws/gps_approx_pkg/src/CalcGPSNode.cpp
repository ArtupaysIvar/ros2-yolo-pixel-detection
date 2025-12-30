#include "pixel_msgs/msg/pixel_coordinates.hpp"
#include <rclcpp/rclcpp.hpp>
#include "geometry_msgs/msg/point.hpp"


class CalcGPSNode : public rclcpp::Node
{   
public:
    CalcGPSNode() : Node("calc_gps_node")
    {
        RCLCPP_INFO(this->get_logger(), "GPS approx. node started");
        
        subscriber_ = this->create_subscription<pixel_msgs::msg::PixelCoordinates>
        ("pixel_topic", 10, std::bind(&CalcGPSNode::pixelCallback, this, 
        std::placeholders::_1));
                
        publisher_ = this->create_publisher<geometry_msgs::msg::Point>
        ("pixel_topic", 10);

        timer_ =  this->create_wall_timer(std::chrono::milliseconds(500), 
        std::bind(&CalcGPSNode::timer_callback), this);
         
    }
private:
    void pixelCallback(const pixel_msgs::msg::PixelCoordinates::SharedPtr YOLOmsg){
        YOLOmsg->u;
        YOLOmsg->v;
        YOLOmsg->confidence;
    }
    void timer_callback(){
        geometry_msgs::msg::Point GPSmsg;
        GPSmsg.x = ;
        GPSmsg.y = ;
        GPSmsg.z = ;

        publisher_->publish(GPSmsg);

        RCLCPP_INFO(
            this->get_logger(),
            "Published Point: x=%.2f y=%.2f z=%.2f",
            GPSmsg.x, GPSmsg.y, GPSmsg.z
        );

    }
    
    rclcpp::Subscription<pixel_msgs::msg::PixelCoordinates>::SharedPtr subscriber_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CalcGPSNode>());
    rclcpp::shutdown();
    return 0;
}
