// Code generated by protoc-gen-go-grpc. DO NOT EDIT.

package name

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

// HelloClient is the client API for Hello service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type HelloClient interface {
	ImageManupilate(ctx context.Context, in *ImgMessage, opts ...grpc.CallOption) (*ImgResponse, error)
}

type helloClient struct {
	cc grpc.ClientConnInterface
}

func NewHelloClient(cc grpc.ClientConnInterface) HelloClient {
	return &helloClient{cc}
}

func (c *helloClient) ImageManupilate(ctx context.Context, in *ImgMessage, opts ...grpc.CallOption) (*ImgResponse, error) {
	out := new(ImgResponse)
	err := c.cc.Invoke(ctx, "/Hello/ImageManupilate", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// HelloServer is the server API for Hello service.
// All implementations must embed UnimplementedHelloServer
// for forward compatibility
type HelloServer interface {
	ImageManupilate(context.Context, *ImgMessage) (*ImgResponse, error)
	mustEmbedUnimplementedHelloServer()
}

// UnimplementedHelloServer must be embedded to have forward compatible implementations.
type UnimplementedHelloServer struct {
}

func (UnimplementedHelloServer) ImageManupilate(context.Context, *ImgMessage) (*ImgResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method ImageManupilate not implemented")
}
func (UnimplementedHelloServer) mustEmbedUnimplementedHelloServer() {}

// UnsafeHelloServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to HelloServer will
// result in compilation errors.
type UnsafeHelloServer interface {
	mustEmbedUnimplementedHelloServer()
}

func RegisterHelloServer(s grpc.ServiceRegistrar, srv HelloServer) {
	s.RegisterService(&Hello_ServiceDesc, srv)
}

func _Hello_ImageManupilate_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ImgMessage)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(HelloServer).ImageManupilate(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/Hello/ImageManupilate",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(HelloServer).ImageManupilate(ctx, req.(*ImgMessage))
	}
	return interceptor(ctx, in, info, handler)
}

// Hello_ServiceDesc is the grpc.ServiceDesc for Hello service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Hello_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "Hello",
	HandlerType: (*HelloServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "ImageManupilate",
			Handler:    _Hello_ImageManupilate_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "protos/hello.proto",
}
