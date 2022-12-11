package com.everyson.mongri_backend.domain.chat.service;

import com.everyson.mongri_backend.domain.chat.model.ChatRoom;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.LinkedHashMap;
import java.util.Map;

@Service
public class ChatRoomService {

    public String createRoom() {
        ChatRoom chatRoom = ChatRoom.create();

        return chatRoom.getRoomId();
    }
}
